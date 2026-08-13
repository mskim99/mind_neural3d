#!/usr/bin/env python3
"""
Visualize the architecture / training connection graph used by train_neural3d_pp.py.

Outputs (default: ./architecture_viz):
  1) training_pipeline.png / .pdf
     - High-level training flow reconstructed from train_neural3d_pp.py.

  2) model_module_tree.png / .pdf
     - Static nn.Module parent-child hierarchy.
     - Does NOT require a forward pass after model construction.

  3) model_forward_graph.png / .pdf
     - PNG: compact runtime overview.
     - PDF: logical MULTI-PAGE runtime architecture.
       Page 1 overview, then major executed modules (encoder/UNet/VAE/etc.),
       then the training loss page.
     - Built from an actual get_loss(batch) execution using forward hooks.

Optional:
  4) autograd_graph.png / .pdf
     - Low-level autograd graph via torchviz. Can become extremely large.

Install:
    pip install graphviz torchviz omegaconf pypdf

Graphviz system binary is also required:
    Ubuntu/Debian: sudo apt-get install graphviz

Example:
    python visualize_neural3d_architecture.py \
        --config ./configs/mind3d.yaml \
        --data_path /data/jionkim/neuro_3D/ \
        --sub_id 0001 \
        --device cuda \
        --out_dir ./architecture_viz

For only lightweight/static visualizations:
    python visualize_neural3d_architecture.py --skip_forward

For autograd visualization as well:
    python visualize_neural3d_architecture.py --with_torchviz
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


# ---------------------------------------------------------------------
# Make project-local `src` importable when this script is placed
# in the repository root.
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.mvdiffusion_var import MVDiffusion
from src.data.egg_dataset import AllDataFeatureTwoEEG


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def render_graphviz_png(dot, out_dir: Path, stem: str, dpi: str = "120"):
    """Render a compact PNG summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        dot.graph_attr.update(dpi=dpi)
    except Exception:
        pass

    rendered = dot.render(
        filename=stem,
        directory=str(out_dir),
        format="png",
        cleanup=True,
    )
    print(f"[saved] {rendered}")
    return Path(rendered)


def render_graphviz_pdf(dot, out_dir: Path, stem: str):
    """Render a one-page PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = dot.render(
        filename=stem,
        directory=str(out_dir),
        format="pdf",
        cleanup=True,
    )
    print(f"[saved] {rendered}")
    return Path(rendered)


def count_params(module: nn.Module, recurse: bool = True):
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def fmt_params(n: int):
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def recursive_to_device(obj, device):
    """
    Recursively move tensors to the requested device while preserving
    dict/list/tuple structure.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [recursive_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(recursive_to_device(v, device) for v in obj)

    return obj


def _render_pdf_page(dot, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = dot.render(
        filename=path.stem,
        directory=str(path.parent),
        format="pdf",
        cleanup=True,
    )
    return Path(rendered)


def _merge_pdfs(pdf_paths, output_path: Path):
    PdfReader = None
    PdfWriter = None

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError:
            pass

    if PdfReader is None:
        raise ImportError(
            "Merging multi-page PDFs requires `pypdf` (recommended).\n"
            "Install it with: pip install pypdf"
        )

    writer = PdfWriter()
    for pdf in pdf_paths:
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)


def _cleanup_temp_pages(page_dir: Path, page_pdfs):
    for p in page_pdfs:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        page_dir.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------
# 1. High-level training pipeline
# ---------------------------------------------------------------------
def build_training_pipeline_overview_graph():
    from graphviz import Digraph

    dot = Digraph("training_pipeline_overview")
    dot.attr(
        rankdir="LR",
        splines="ortho",
        nodesep="0.35",
        ranksep="0.55",
        bgcolor="white",
        label="Training pipeline overview",
        labelloc="t",
        fontsize="20",
        size="11.0,7.5!",
        ratio="compress",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F7F7F7",
        color="#555555",
        fontname="Helvetica",
        fontsize="10",
        margin="0.12,0.08",
    )
    dot.attr(
        "edge",
        color="#555555",
        fontname="Helvetica",
        fontsize="9",
        arrowsize="0.7",
    )

    nodes = {
        "dataset": "AllDataFeatureTwoEEG\nnum_frames=16",
        "loader": "DataLoader\nbatch = train_item",
        "model": "MVDiffusion\n(src.mvdiffusion_var)",
        "get_loss": "model_full.get_loss(train_item)",
        "diff": "diff_loss\nweight = 1.0",
        "clip": "clip_loss\nweight = 0.5",
        "ortho": "ortho_loss\nweight = 0.1",
        "sum": "weighted total loss",
        "accum": "loss / accumulation_steps",
        "backward": "loss.backward()",
        "optim": "model_full.opt.step()",
        "sched": "model_full.sche.step()",
        "vis": "Visualization branch",
    }

    for key, label in nodes.items():
        dot.node(key, label)

    dot.edge("dataset", "loader")
    dot.edge("loader", "get_loss", label="train_item")
    dot.edge("model", "get_loss")

    dot.edge("get_loss", "diff")
    dot.edge("get_loss", "clip")
    dot.edge("get_loss", "ortho")

    dot.edge("diff", "sum")
    dot.edge("clip", "sum")
    dot.edge("ortho", "sum")

    dot.edge("sum", "accum")
    dot.edge("accum", "backward")
    dot.edge("backward", "optim")
    dot.edge("optim", "sched")
    dot.edge("model", "vis", style="dashed", label="visualize_every")

    return dot


def build_training_visualization_graph():
    from graphviz import Digraph

    dot = Digraph("training_visualization")
    dot.attr(
        rankdir="LR",
        splines="ortho",
        nodesep="0.35",
        ranksep="0.55",
        bgcolor="white",
        label="Visualization / inference branch",
        labelloc="t",
        fontsize="20",
        size="11.0,7.5!",
        ratio="compress",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F8F8F8",
        color="#555555",
        fontname="Helvetica",
        fontsize="10",
        margin="0.12,0.08",
    )
    dot.attr("edge", color="#555555", fontname="Helvetica", fontsize="9", arrowsize="0.7")

    nodes = {
        "test_loader": "test_loader / train_item",
        "prepare": "prepare_batch_data(batch)",
        "enc_cond": "encode_embed_fmri_condition_fmri(cond_eeg)",
        "enc_uncond": "encode_embed_fmri_condition_fmri(zeros_like(cond_eeg))",
        "concat": "concat cond/uncond\nprompt_embeds + eeg_cond_latents",
        "scheduler": "scheduler.set_timesteps(50)",
        "noise": "initial latent noise\n(B, 4, 120, 80)",
        "loop": "manual denoising loop\nforward_unet + CFG(4.0)",
        "vae": "pipeline.vae.decode(...)",
        "norm": "renormalize/clamp\n[0, 1]",
        "grid": "make_grid(target_imgs, images_pred)",
        "save": "save_image(...)",
    }

    for k, v in nodes.items():
        dot.node(k, v)

    dot.edge("test_loader", "prepare")
    dot.edge("prepare", "enc_cond")
    dot.edge("prepare", "enc_uncond")
    dot.edge("enc_cond", "concat")
    dot.edge("enc_uncond", "concat")
    dot.edge("concat", "scheduler")
    dot.edge("scheduler", "noise")
    dot.edge("noise", "loop")
    dot.edge("concat", "loop")
    dot.edge("loop", "vae")
    dot.edge("vae", "norm")
    dot.edge("norm", "grid")
    dot.edge("grid", "save")

    return dot


def render_training_pipeline_artifacts(out_dir: Path):
    """
    training_pipeline.png  -> overview only
    training_pipeline.pdf  -> multi-page:
        1) training overview
        2) visualization / inference branch
    """
    overview = build_training_pipeline_overview_graph()
    detail = build_training_visualization_graph()

    render_graphviz_png(overview, out_dir, "training_pipeline")

    page_dir = out_dir / ".training_pipeline_pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    page_pdfs = [
        _render_pdf_page(overview, page_dir / "01_training_overview.pdf"),
        _render_pdf_page(detail, page_dir / "02_visualization_branch.pdf"),
    ]

    final_pdf = out_dir / "training_pipeline.pdf"
    _merge_pdfs(page_pdfs, final_pdf)
    print(f"[saved] {final_pdf}")
    print(f"[saved] {len(page_pdfs)} logical PDF pages")
    _cleanup_temp_pages(page_dir, page_pdfs)


# ---------------------------------------------------------------------
# 2. Static module hierarchy graph
# ---------------------------------------------------------------------
def build_module_tree_graph(model: nn.Module, max_depth: int = 4):
    """
    Visualize nn.Module containment without executing forward().
    Useful when torchview tracing fails due to dynamic control flow,
    custom CUDA ops, diffusion schedulers, etc.
    """
    from graphviz import Digraph

    dot = Digraph("model_module_tree")
    dot.attr(
        rankdir="TB",
        compound="true",
        splines="polyline",
        nodesep="0.18",
        ranksep="0.35",
        bgcolor="white",
        label=f"MVDiffusion module hierarchy (depth <= {max_depth})",
        labelloc="t",
        fontsize="20",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F8F8F8",
        color="#555555",
        fontname="Helvetica",
        fontsize="9",
        margin="0.10,0.06",
    )
    dot.attr(
        "edge",
        color="#777777",
        arrowsize="0.55",
    )

    seen = set()

    def add_module(module: nn.Module, path: str, depth: int):
        node_id = path if path else "__root__"
        if node_id in seen:
            return
        seen.add(node_id)

        class_name = module.__class__.__name__
        total_p = count_params(module, recurse=True)
        own_p = count_params(module, recurse=False)

        display_path = path if path else "MVDiffusion"
        label = (
            f"{display_path}\n"
            f"{class_name}\n"
            f"params={fmt_params(total_p)}"
        )
        if own_p and own_p != total_p:
            label += f" (own {fmt_params(own_p)})"

        dot.node(node_id, label)

        if depth >= max_depth:
            children = list(module.named_children())
            if children:
                hidden_id = f"{node_id}.__hidden__"
                dot.node(
                    hidden_id,
                    f"... {len(children)} child module(s) hidden ...",
                    shape="plaintext",
                )
                dot.edge(node_id, hidden_id, style="dashed")
            return

        for child_name, child in module.named_children():
            child_path = f"{path}.{child_name}" if path else child_name
            child_id = child_path
            add_module(child, child_path, depth + 1)
            dot.edge(node_id, child_id, label=child_name)

    add_module(model, "", 0)
    return dot


def _build_module_tree_page(model: nn.Module, root_name: str, max_relative_depth: int = 3, title: str = None):
    """
    One readable page of the static nn.Module hierarchy.
    root_name == "" means the global/root overview page.
    """
    from graphviz import Digraph

    dot = Digraph("module_tree_page_" + (root_name.replace(".", "_") or "root"))
    dot.attr(
        rankdir="TB",
        compound="true",
        splines="polyline",
        nodesep="0.18",
        ranksep="0.34",
        bgcolor="white",
        label=title or ("MVDiffusion module hierarchy" if root_name == "" else f"{root_name} module hierarchy"),
        labelloc="t",
        fontsize="18",
        size="11.0,7.5!",
        ratio="compress",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F8F8F8",
        color="#555555",
        fontname="Helvetica",
        fontsize="8.5",
        margin="0.08,0.05",
    )
    dot.attr("edge", color="#777777", arrowsize="0.5")

    modules = dict(model.named_modules())

    if root_name == "":
        root_module = model
        prefix = ""
    else:
        root_module = modules[root_name]
        prefix = root_name

    shown = []

    for name, mod in modules.items():
        if root_name == "":
            # Root overview page: only root + top-level children + one level below.
            if name == "" or name.count(".") <= max_relative_depth - 1:
                shown.append(name)
        else:
            if name == root_name or name.startswith(root_name + "."):
                rel = name[len(root_name):].lstrip(".")
                depth = 0 if name == root_name else rel.count(".") + 1
                if depth <= max_relative_depth:
                    shown.append(name)

    if root_name != "" and root_name not in shown:
        shown.append(root_name)

    shown = sorted(shown, key=lambda n: (n.count("."), n))
    shown_set = set(shown)

    def node_id(name):
        return "mt_" + ("root" if name == "" else "".join(c if c.isalnum() else "_" for c in name))

    for name in shown:
        mod = model if name == "" else modules[name]
        class_name = mod.__class__.__name__
        total_p = count_params(mod, recurse=True)
        own_p = count_params(mod, recurse=False)
        display = "MVDiffusion" if name == "" else name

        label = f"{display}\n{class_name}\nparams={fmt_params(total_p)}"
        if own_p and own_p != total_p:
            label += f" (own {fmt_params(own_p)})"

        dot.node(node_id(name), label)

    for name in shown:
        if name == "":
            continue

        if "." in name:
            parent = name.rsplit(".", 1)[0]
        else:
            parent = ""

        # walk up to nearest shown parent
        while parent not in shown_set:
            if not parent:
                break
            parent = parent.rsplit(".", 1)[0] if "." in parent else ""

        if parent in shown_set:
            edge_label = name.split(".")[-1]
            dot.edge(node_id(parent), node_id(name), label=edge_label)

    # hidden deeper descendants notice
    if root_name != "":
        deeper = []
        for name, _ in modules.items():
            if name.startswith(root_name + "."):
                rel = name[len(root_name):].lstrip(".")
                depth = rel.count(".") + 1
                if depth > max_relative_depth:
                    deeper.append(name)
        if deeper:
            hid = node_id(root_name + ".__more__")
            dot.node(hid, f"... {len(deeper)} deeper child module(s) omitted ...", shape="plaintext")
            dot.edge(node_id(root_name), hid, style="dashed")

    return dot


def render_module_tree_artifacts(model: nn.Module, out_dir: Path, max_depth: int = 4):
    """
    model_module_tree.png -> compact root overview
    model_module_tree.pdf -> multi-page:
        1) root/top-level overview
        2+) one page per top-level child module
    """
    root_page_depth = min(2, max_depth)
    overview = _build_module_tree_page(
        model,
        root_name="",
        max_relative_depth=root_page_depth,
        title="MVDiffusion module hierarchy overview",
    )

    render_graphviz_png(overview, out_dir, "model_module_tree")

    modules = dict(model.named_modules())
    top_level = [name for name, _ in model.named_children()]

    page_dir = out_dir / ".model_module_tree_pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    page_pdfs = [
        _render_pdf_page(overview, page_dir / "01_root_overview.pdf")
    ]

    page_idx = 2
    for name in top_level:
        child_page = _build_module_tree_page(
            model,
            root_name=name,
            max_relative_depth=max_depth,
            title=f"{name} module hierarchy",
        )
        page_pdfs.append(
            _render_pdf_page(child_page, page_dir / f"{page_idx:02d}_{name.replace('.', '_')}.pdf")
        )
        page_idx += 1

    final_pdf = out_dir / "model_module_tree.pdf"
    _merge_pdfs(page_pdfs, final_pdf)
    print(f"[saved] {final_pdf}")
    print(f"[saved] {len(page_pdfs)} logical PDF pages")
    _cleanup_temp_pages(page_dir, page_pdfs)


# ---------------------------------------------------------------------
# 3. Runtime get_loss graph using torchview
# ---------------------------------------------------------------------
class TrainingLossWrapper(nn.Module):
    """
    Wrap the training path as a normal forward():

        batch
          -> MVDiffusion.get_loss(batch)
          -> diff_loss, clip_loss, ortho_loss
          -> 1.0*diff + 0.5*clip + 0.1*ortho

    This mirrors the active loss path in train_neural3d_pp.py.
    """
    def __init__(
        self,
        model: nn.Module,
        lambda_diff: float = 1.0,
        lambda_clip: float = 0.5,
        lambda_ortho: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.lambda_diff = lambda_diff
        self.lambda_clip = lambda_clip
        self.lambda_ortho = lambda_ortho

    def forward(self, batch):
        diff_loss, clip_loss, ortho_loss = self.model.get_loss(batch)

        total_loss = (
            self.lambda_diff * diff_loss
            + self.lambda_clip * clip_loss
            + self.lambda_ortho * ortho_loss
        )

        # A single tensor output keeps visualization relatively clean.
        return total_loss


def _shape_summary(obj, max_items=3):
    """Compact tensor/output shape summary for graph labels."""
    if torch.is_tensor(obj):
        return str(tuple(obj.shape))

    if isinstance(obj, (list, tuple)):
        vals = []
        for x in obj[:max_items]:
            vals.append(_shape_summary(x, max_items=max_items))
        if len(obj) > max_items:
            vals.append("...")
        return "[" + ", ".join(vals) + "]"

    if isinstance(obj, dict):
        vals = []
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                vals.append("...")
                break
            vals.append(f"{k}:{_shape_summary(v, max_items=max_items)}")
        return "{" + ", ".join(vals) + "}"

    return type(obj).__name__


def _safe_node_id(name: str):
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)


class RuntimeModuleRecorder:
    """
    Record which nn.Modules are actually executed during get_loss(batch).

    This avoids a single gigantic torchview graph.  We still use a real
    training batch and real forward execution, but summarize the execution
    module-by-module into logical pages.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles = []
        self.events = []
        self.stats = {}
        self.module_lookup = dict(model.named_modules())

    def _hook(self, name):
        def fn(module, inputs, output):
            event = {
                "name": name,
                "class": module.__class__.__name__,
                "input": _shape_summary(inputs),
                "output": _shape_summary(output),
            }
            self.events.append(event)

            stat = self.stats.setdefault(
                name,
                {
                    "class": module.__class__.__name__,
                    "calls": 0,
                    "first_index": len(self.events) - 1,
                    "input": event["input"],
                    "output": event["output"],
                },
            )
            stat["calls"] += 1
            stat["output"] = event["output"]

        return fn

    def install(self):
        # Skip root "" because we want named components.
        for name, module in self.model.named_modules():
            if not name:
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def run(self, wrapper, sample_batch):
        self.install()
        try:
            wrapper.eval()
            # Do not build a huge autograd graph just for page construction.
            with torch.no_grad():
                loss = wrapper(sample_batch)
            return loss
        finally:
            self.remove()


def _relative_depth(full_name: str, prefix: str):
    if full_name == prefix:
        return 0
    suffix = full_name[len(prefix):].lstrip(".")
    if not suffix:
        return 0
    return suffix.count(".") + 1


def _find_runtime_roots(recorder: RuntimeModuleRecorder):
    """
    Find meaningful runtime component roots dynamically.

    Preferred semantic groups:
      fmri/eeg encoder -> UNet -> VAE
    Then include other actually-used top-level roots not covered by them.
    """
    used = set(recorder.stats.keys())
    all_names = list(recorder.module_lookup.keys())

    def candidates_by_token(tokens):
        out = []
        for name in all_names:
            lname = name.lower()
            leaf = lname.split(".")[-1]
            if any(
                leaf == tok
                or leaf.endswith("_" + tok)
                or tok in leaf
                for tok in tokens
            ):
                if name in used or any(u.startswith(name + ".") for u in used):
                    out.append(name)

        # Prefer the shallowest path; then shortest spelling.
        out.sort(key=lambda x: (x.count("."), len(x)))
        return out

    semantic = []

    fmri = candidates_by_token(["fmri_encoder", "eeg_encoder"])
    if fmri:
        semantic.append(("Condition encoder", fmri[0]))

    unet = candidates_by_token(["unet"])
    if unet:
        semantic.append(("Diffusion UNet", unet[0]))

    vae = candidates_by_token(["vae"])
    if vae:
        semantic.append(("VAE", vae[0]))

    # Add other used top-level modules that are not already inside a semantic root.
    covered_roots = [prefix for _, prefix in semantic]

    top_level_used = []
    for name, stat in recorder.stats.items():
        top = name.split(".")[0]
        if top not in top_level_used:
            top_level_used.append(top)

    for top in top_level_used:
        if any(
            top == root
            or top.startswith(root + ".")
            or root.startswith(top + ".")
            for root in covered_roots
        ):
            continue

        if top in recorder.module_lookup:
            semantic.append((top, top))

    # Sort by actual first execution rather than by attribute name.
    def first_call(item):
        _, prefix = item
        indices = [
            st["first_index"]
            for name, st in recorder.stats.items()
            if name == prefix or name.startswith(prefix + ".")
        ]
        return min(indices) if indices else 10**12

    semantic.sort(key=first_call)
    return semantic


def _build_runtime_overview_graph(
    recorder: RuntimeModuleRecorder,
    roots,
    loss_value=None,
):
    from graphviz import Digraph

    dot = Digraph("runtime_overview")
    dot.attr(
        rankdir="LR",
        splines="ortho",
        nodesep="0.45",
        ranksep="0.65",
        bgcolor="white",
        label="MVDiffusion runtime overview - actual get_loss(batch) execution",
        labelloc="t",
        fontsize="18",
        size="11.0,7.5!",
        ratio="compress",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F7F7F7",
        color="#555555",
        fontname="Helvetica",
        fontsize="10",
        margin="0.14,0.09",
    )
    dot.attr(
        "edge",
        color="#555555",
        fontname="Helvetica",
        fontsize="9",
        arrowsize="0.7",
    )

    dot.node("batch", "Training batch\nAllDataFeatureTwoEEG")

    prev = "batch"

    for i, (title, prefix) in enumerate(roots):
        related = [
            (name, st)
            for name, st in recorder.stats.items()
            if name == prefix or name.startswith(prefix + ".")
        ]
        calls = sum(st["calls"] for _, st in related)

        module = recorder.module_lookup.get(prefix)
        params = count_params(module) if module is not None else 0

        # Use the root call shape if present, otherwise first child.
        stat = recorder.stats.get(prefix)
        if stat is None and related:
            related_sorted = sorted(related, key=lambda x: x[1]["first_index"])
            stat = related_sorted[0][1]

        shape_txt = ""
        if stat is not None:
            shape_txt = f"\nout: {stat['output']}"

        nid = f"group_{i}"
        dot.node(
            nid,
            f"{title}\n{prefix}\n"
            f"params={fmt_params(params)} | runtime calls={calls}"
            f"{shape_txt}",
        )
        dot.edge(prev, nid)
        prev = nid

    dot.node(
        "loss",
        "TrainingLossWrapper\n"
        "get_loss(batch)\n"
        "1.0*diff + 0.5*clip + 0.1*ortho",
    )
    dot.edge(prev, "loss")

    if loss_value is not None:
        try:
            lv = float(loss_value.detach().cpu())
            dot.node("total", f"total_loss\n{lv:.5g}")
        except Exception:
            dot.node("total", "total_loss")
    else:
        dot.node("total", "total_loss")

    dot.edge("loss", "total")
    return dot


def _build_runtime_component_graph(
    recorder: RuntimeModuleRecorder,
    title: str,
    prefix: str,
    max_relative_depth: int = 4,
):
    """
    Build one readable page for a runtime component.

    Only modules that ACTUALLY executed are shown.
    The hierarchy is used for edges so the page remains interpretable and
    avoids the thousands of tensor-operation nodes produced by torchview.
    """
    from graphviz import Digraph

    dot = Digraph(_safe_node_id(prefix))
    dot.attr(
        rankdir="TB",
        splines="polyline",
        nodesep="0.20",
        ranksep="0.32",
        bgcolor="white",
        label=f"{title} - runtime module calls",
        labelloc="t",
        fontsize="18",
        size="11.0,7.5!",
        ratio="compress",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F8F8F8",
        color="#555555",
        fontname="Helvetica",
        fontsize="8.5",
        margin="0.08,0.05",
    )
    dot.attr(
        "edge",
        color="#777777",
        arrowsize="0.5",
    )

    used_names = {
        name
        for name in recorder.stats
        if (
            name == prefix or name.startswith(prefix + ".")
        )
        and _relative_depth(name, prefix) <= max_relative_depth
    }

    # If prefix itself is a container that never directly executes,
    # add it as a synthetic root.
    synthetic_root = prefix not in used_names
    if synthetic_root:
        used_names.add(prefix)

    if not used_names:
        dot.node(
            "empty",
            f"No recorded runtime calls under\n{prefix}",
            shape="plaintext",
        )
        return dot

    def nearest_visible_parent(name):
        if name == prefix:
            return None

        parent = name.rsplit(".", 1)[0] if "." in name else ""
        while parent:
            if parent in used_names:
                return parent
            if parent == prefix:
                return prefix
            parent = parent.rsplit(".", 1)[0] if "." in parent else ""

        return prefix if prefix in used_names else None

    ordered = sorted(
        used_names,
        key=lambda n: (
            _relative_depth(n, prefix),
            recorder.stats.get(n, {}).get("first_index", 10**12),
            n,
        ),
    )

    for name in ordered:
        nid = _safe_node_id(name)

        module = recorder.module_lookup.get(name)
        class_name = module.__class__.__name__ if module is not None else "ModuleGroup"
        params = count_params(module) if module is not None else 0

        stat = recorder.stats.get(name)
        if stat is None:
            label = (
                f"{name}\n{class_name}\n"
                f"params={fmt_params(params)}"
            )
        else:
            label = (
                f"{name}\n{stat['class']}\n"
                f"params={fmt_params(params)} | calls={stat['calls']}\n"
                f"out: {stat['output']}"
            )

        dot.node(nid, label)

    for name in ordered:
        parent = nearest_visible_parent(name)
        if parent is not None and parent in used_names:
            dot.edge(_safe_node_id(parent), _safe_node_id(name))

    # Indicate hidden deeper runtime calls.
    hidden = [
        name
        for name in recorder.stats
        if (name == prefix or name.startswith(prefix + "."))
        and _relative_depth(name, prefix) > max_relative_depth
    ]
    if hidden:
        hid = _safe_node_id(prefix + ".__more__")
        dot.node(
            hid,
            f"... {len(hidden)} deeper executed module(s) omitted ...",
            shape="plaintext",
        )
        dot.edge(_safe_node_id(prefix), hid, style="dashed")

    return dot


def _build_loss_page():
    from graphviz import Digraph

    dot = Digraph("loss_composition")
    dot.attr(
        rankdir="LR",
        splines="ortho",
        nodesep="0.5",
        ranksep="0.7",
        bgcolor="white",
        label="Training loss composition",
        labelloc="t",
        fontsize="18",
        size="11.0,7.5!",
        ratio="compress",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F8F8F8",
        color="#555555",
        fontname="Helvetica",
        fontsize="10",
    )
    dot.attr("edge", color="#666666", arrowsize="0.7")

    dot.node("getloss", "MVDiffusion.get_loss(batch)")
    dot.node("diff", "diff_loss\nweight 1.0")
    dot.node("clip", "clip_loss\nweight 0.5")
    dot.node("ortho", "ortho_loss\nweight 0.1")
    dot.node(
        "sum",
        "total_loss =\n"
        "1.0 * diff_loss\n"
        "+ 0.5 * clip_loss\n"
        "+ 0.1 * ortho_loss",
    )
    dot.node("backward", "loss / accumulation_steps\n-> backward()")

    dot.edge("getloss", "diff")
    dot.edge("getloss", "clip")
    dot.edge("getloss", "ortho")
    dot.edge("diff", "sum")
    dot.edge("clip", "sum")
    dot.edge("ortho", "sum")
    dot.edge("sum", "backward")

    return dot


def _render_pdf_page(dot, path: Path):
    """
    Render a graph as one bounded landscape page.

    Graphviz 'size' above keeps each logical graph on one page rather than
    producing a physically tiled/cropped mega-graph.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rendered = dot.render(
        filename=path.stem,
        directory=str(path.parent),
        format="pdf",
        cleanup=True,
    )
    return Path(rendered)


def _merge_pdfs(pdf_paths, output_path: Path):
    """
    Merge logical page PDFs into one multi-page model_forward_graph.pdf.

    Tries pypdf, then PyPDF2.  Gives a clear install message if neither exists.
    """
    PdfReader = None
    PdfWriter = None

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError:
            pass

    if PdfReader is None:
        raise ImportError(
            "Multi-page PDF merge requires `pypdf` (recommended).\n"
            "Install it with: pip install pypdf"
        )

    writer = PdfWriter()
    for pdf in pdf_paths:
        reader = PdfReader(str(pdf))
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)


def render_torchview_graph(
    wrapper: nn.Module,
    sample_batch,
    out_dir: Path,
    depth: int = 4,
    device: str = "cuda",
):
    """
    NEW behavior:
      model_forward_graph.pdf is a logical multi-page PDF, not one gigantic page.

    Pages are grounded in an ACTUAL get_loss(batch) execution recorded with
    PyTorch forward hooks.

      Page 1: runtime overview
      Page 2+: major runtime components (fmri/eeg encoder, UNet, VAE, etc.)
      Final page: loss composition

    `depth` now controls the maximum relative module depth shown INSIDE each
    component page.
    """
    print("[runtime] recording actual get_loss(batch) module calls ...")

    recorder = RuntimeModuleRecorder(wrapper.model)
    loss_value = recorder.run(wrapper, sample_batch)

    roots = _find_runtime_roots(recorder)

    print(f"[runtime] recorded {len(recorder.events)} module calls")
    print(f"[runtime] unique executed modules: {len(recorder.stats)}")
    print("[runtime] logical pages:")
    for title, prefix in roots:
        print(f"  - {title}: {prefix}")

    page_dir = out_dir / ".model_forward_graph_pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    page_pdfs = []

    # Page 1: overview.
    overview = _build_runtime_overview_graph(
        recorder,
        roots,
        loss_value=loss_value,
    )
    page_pdfs.append(
        _render_pdf_page(
            overview,
            page_dir / "01_overview.pdf",
        )
    )

    # Also save a compact PNG overview as model_forward_graph.png.
    overview.graph_attr.update(dpi="120")
    png = overview.render(
        filename="model_forward_graph",
        directory=str(out_dir),
        format="png",
        cleanup=True,
    )
    print(f"[saved] {png}")

    # Component pages.
    page_idx = 2
    for title, prefix in roots:
        graph = _build_runtime_component_graph(
            recorder,
            title=title,
            prefix=prefix,
            max_relative_depth=depth,
        )
        page_pdfs.append(
            _render_pdf_page(
                graph,
                page_dir / f"{page_idx:02d}_{_safe_node_id(prefix)}.pdf",
            )
        )
        page_idx += 1

    # Final page: training loss composition.
    loss_page = _build_loss_page()
    page_pdfs.append(
        _render_pdf_page(
            loss_page,
            page_dir / f"{page_idx:02d}_loss.pdf",
        )
    )

    final_pdf = out_dir / "model_forward_graph.pdf"
    _merge_pdfs(page_pdfs, final_pdf)

    print(f"[saved] {final_pdf}")
    print(f"[saved] {len(page_pdfs)} logical PDF pages")

    # Remove intermediate one-page PDFs.  Keep only the final user-facing PDF.
    for p in page_pdfs:
        try:
            p.unlink()
        except OSError:
            pass

    try:
        page_dir.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------
# 4. Optional low-level autograd graph
# ---------------------------------------------------------------------
def render_torchviz_graph(
    wrapper: nn.Module,
    sample_batch,
    out_dir: Path,
):
    """
    Very detailed gradient-level graph.
    Warning: Stable Diffusion / UNet graphs can be huge.
    """
    from torchviz import make_dot

    print("[torchviz] building autograd graph (this may be very large) ...")

    wrapper.train()

    total_loss = wrapper(sample_batch)

    dot = make_dot(
        total_loss,
        params=dict(wrapper.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )

    render_graphviz(dot, out_dir, "autograd_graph")


# ---------------------------------------------------------------------
# Model / data construction
# ---------------------------------------------------------------------
def _find_conditioning_configs(cfg):
    """
    Find a mapping that contains both:
      - stable_diffusion_config
      - fmri_encoder_config

    The original training script accesses:
        cfg.model.params.stable_diffusion_config
        cfg.model.params.fmri_encoder_config

    However, visualization should fail gracefully if the selected YAML has a
    slightly different nesting structure or if the wrong YAML was selected.
    """
    container = OmegaConf.to_container(cfg, resolve=False)

    matches = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            if (
                "stable_diffusion_config" in obj
                and "fmri_encoder_config" in obj
            ):
                matches.append((path, obj))

            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                walk(value, child_path)

        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                child_path = f"{path}[{i}]"
                walk(value, child_path)

    walk(container)

    if not matches:
        return None, None, None

    # Prefer the exact path used by train_neural3d_pp.py when available.
    for path, mapping in matches:
        if path == "model.params":
            return (
                OmegaConf.create(mapping["stable_diffusion_config"]),
                OmegaConf.create(mapping["fmri_encoder_config"]),
                path,
            )

    # Otherwise use the first paired occurrence and report where it was found.
    path, mapping = matches[0]
    return (
        OmegaConf.create(mapping["stable_diffusion_config"]),
        OmegaConf.create(mapping["fmri_encoder_config"]),
        path,
    )


def build_model(args, device):
    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    print(f"[config] loading: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Print keys immediately so a wrong config file is obvious.
    if OmegaConf.is_dict(cfg):
        top_keys = list(cfg.keys())
    else:
        top_keys = []

    print(f"[config] top-level keys: {top_keys}")

    # First try the exact structure used by train_neural3d_pp.py.
    stable_cfg = OmegaConf.select(
        cfg,
        "model.params.stable_diffusion_config",
        default=None,
    )
    fmri_cfg = OmegaConf.select(
        cfg,
        "model.params.fmri_encoder_config",
        default=None,
    )

    found_path = "model.params"

    # If the selected YAML is nested differently, search recursively.
    if stable_cfg is None or fmri_cfg is None:
        stable_cfg, fmri_cfg, found_path = _find_conditioning_configs(cfg)

    if stable_cfg is None or fmri_cfg is None:
        raise KeyError(
            "\nCould not find both `stable_diffusion_config` and "
            "`fmri_encoder_config` in the selected YAML.\n\n"
            f"Loaded config : {config_path}\n"
            f"Top-level keys: {top_keys}\n\n"
            "The provided training script expects these values at:\n"
            "  cfg.model.params.stable_diffusion_config\n"
            "  cfg.model.params.fmri_encoder_config\n\n"
            "Most likely causes:\n"
            "  1) --config points to a different YAML than the training run,\n"
            "  2) the YAML uses a different nesting structure, or\n"
            "  3) this YAML is only a partial config.\n\n"
            "Run the training command with the SAME --config path you use here."
        )

    print(f"[config] conditioning configs found under: {found_path}")

    # Same constructor semantics as train_neural3d_pp.py, but robust to YAML nesting.
    model = MVDiffusion(
        cfg,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(args.runtime_logdir),
    )

    model = model.to(device)
    return model, cfg


def get_one_batch(args):
    # Same dataset class / basic settings as train_neural3d_pp.py.
    dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=True,
        num_frames=args.num_frames,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0,        # visualization/debugging: safer than worker processes
        persistent_workers=False,
        drop_last=False,
        shuffle=False,
    )

    return next(iter(loader))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MVDiffusion architecture used by train_neural3d_pp.py"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/mind3d.yaml",
        help="Same OmegaConf YAML used for training.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/jionkim/neuro_3D/",
        help="Dataset root.",
    )
    parser.add_argument("--sub_id", type=str, default="0001")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./architecture_viz"),
    )
    parser.add_argument(
        "--runtime_logdir",
        type=Path,
        default=Path("./architecture_viz/runtime"),
        help="logdir argument passed to MVDiffusion constructor.",
    )

    parser.add_argument(
        "--module_depth",
        type=int,
        default=4,
        help="Maximum depth for static nn.Module tree.",
    )
    parser.add_argument(
        "--forward_depth",
        type=int,
        default=4,
        help=(
            "Maximum relative nn.Module depth shown on each logical "
            "model_forward_graph.pdf component page."
        ),
    )

    parser.add_argument(
        "--skip_forward",
        action="store_true",
        help="Only create training pipeline + module hierarchy.",
    )
    parser.add_argument(
        "--check_config_only",
        action="store_true",
        help="Load/inspect the YAML and construct the model, then exit before visualization.",
    )
    parser.add_argument(
        "--with_torchviz",
        action="store_true",
        help="Also generate a detailed autograd graph.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.runtime_logdir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    # --------------------------------------------------------------
    # A) Guaranteed high-level graph from the training script itself
    # --------------------------------------------------------------
    print("\n[1/3] Rendering high-level training pipeline ...")
    try:
        render_training_pipeline_artifacts(args.out_dir)
    except Exception:
        print("[ERROR] Failed to render training pipeline.")
        traceback.print_exc()

    # --------------------------------------------------------------
    # B) Construct actual model
    # --------------------------------------------------------------
    print("\n[2/3] Constructing MVDiffusion ...")
    model, cfg = build_model(args, device)

    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.check_config_only:
        print("\n[OK] Config resolved and MVDiffusion constructed successfully.")
        return

    print("\nRendering static nn.Module hierarchy ...")
    try:
        render_module_tree_artifacts(
            model,
            args.out_dir,
            max_depth=args.module_depth,
        )
    except Exception:
        print("[ERROR] Failed to render static module tree.")
        traceback.print_exc()

    # --------------------------------------------------------------
    # C) Actual runtime graph
    # --------------------------------------------------------------
    if args.skip_forward:
        print("\n[3/3] Runtime graph skipped (--skip_forward).")
        print(f"\nDone. Results are in: {args.out_dir.resolve()}")
        return

    print("\n[3/3] Loading one real training batch ...")
    sample_batch = get_one_batch(args)
    sample_batch = recursive_to_device(sample_batch, device)

    wrapper = TrainingLossWrapper(model).to(device)

    try:
        render_torchview_graph(
            wrapper=wrapper,
            sample_batch=sample_batch,
            out_dir=args.out_dir,
            depth=args.forward_depth,
            device=args.device,
        )
    except Exception:
        print("\n[ERROR] multi-page runtime graph generation failed.")
        print(
            "The static `model_module_tree.*` and `training_pipeline.*` files "
            "should still be available."
        )
        traceback.print_exc()

    if args.with_torchviz:
        try:
            render_torchviz_graph(
                wrapper=wrapper,
                sample_batch=sample_batch,
                out_dir=args.out_dir,
            )
        except Exception:
            print("\n[ERROR] torchviz graph generation failed.")
            traceback.print_exc()

    print(f"\nDone. Results are in: {args.out_dir.resolve()}")
    print("Expected files:")
    print("  training_pipeline.png     (compact overview)")
    print("  training_pipeline.pdf     (multi-page)")
    print("  model_module_tree.png     (compact root overview)")
    print("  model_module_tree.pdf     (multi-page)")
    print("  model_forward_graph.png   (compact runtime overview)")
    print("  model_forward_graph.pdf   (logical multi-page runtime graph)")
    if args.with_torchviz:
        print("  autograd_graph.png        (if torchviz succeeds)")
        print("  autograd_graph.pdf        (if torchviz succeeds)")


if __name__ == "__main__":
    main()