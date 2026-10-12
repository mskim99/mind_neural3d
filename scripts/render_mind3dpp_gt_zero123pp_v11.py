#!/usr/bin/env python3
"""
Render MinD-3D++ / Objaverse GT .glb meshes into the 6 fixed Zero123++ v1.1 poses.

Run with Blender, NOT normal Python:

    blender -b --python render_mind3dpp_gt_zero123pp_v11.py -- \
        --data_path /data/jionkim/mind3d++ \
        --gt_mesh_root /data/jionkim/GTmeshes/objaverse/glbs \
        --output_dir /data/jionkim/mind3dpp_gt_views

Output:
    <output_dir>/<uid>/00.png
    <output_dir>/<uid>/01.png
    ...
    <output_dir>/<uid>/05.png
    <output_dir>/<uid>/render_meta.json

Zero123++ v1.1 official output camera poses:
    azimuth (relative): 30, 90, 150, 210, 270, 330 deg
    elevation (absolute): 30, -20, 30, -20, 30, -20 deg

Important:
- Zero123++ v1.1 does NOT publish a fixed output FOV in the official README.
  Therefore --fov_deg is a rendering/evaluation choice, not an official v1.1 constant.
- Because the original input-view azimuth is not defined for a standalone Objaverse GT mesh,
  this script defines canonical azimuth 0 as Blender camera position (0, -R, 0)
  looking at the object origin. Use --azimuth_offset_deg if a different canonical
  orientation is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


ZP11_AZIMUTHS = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
ZP11_ELEVATIONS = [30.0, -20.0, 30.0, -20.0, 30.0, -20.0]


def blender_cli_args():
    """Return args after Blender's '--' separator."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def parse_args():
    p = argparse.ArgumentParser(
        description="Render MinD-3D++ Objaverse GT meshes at Zero123++ v1.1 poses."
    )
    p.add_argument(
        "--data_path",
        required=True,
        help="MinD-3D++ root containing test_list.txt.",
    )
    p.add_argument(
        "--gt_mesh_root",
        required=True,
        help="Objaverse GLB root corresponding to paths stored in test_list.txt.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output root. Each UID gets its own subdirectory.",
    )
    p.add_argument(
        "--test_list",
        default="",
        help="Optional explicit test_list.txt path. Default: <data_path>/test_list.txt",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=320,
        help="Square PNG resolution. 320 matches a Zero123++ tile size.",
    )
    p.add_argument(
        "--fov_deg",
        type=float,
        default=30.0,
        help=(
            "Perspective camera FOV used only for GT rendering. "
            "This is NOT an official Zero123++ v1.1 constant."
        ),
    )
    p.add_argument(
        "--azimuth_offset_deg",
        type=float,
        default=0.0,
        help=(
            "Global rotation added to all 6 azimuths. "
            "0 means canonical front camera is at (0,-R,0)."
        ),
    )
    p.add_argument(
        "--object_extent",
        type=float,
        default=2.0,
        help="Normalize the object's maximum bbox extent to this value.",
    )
    p.add_argument(
        "--camera_margin",
        type=float,
        default=1.15,
        help="Extra framing margin applied to the bounding sphere.",
    )
    p.add_argument(
        "--target_fill",
        type=float,
        default=0.65,
        help=(
            "Maximum projected object occupancy across all six views. "
            "A single shared camera radius is fitted for every view."
        ),
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=0,
    )
    p.add_argument(
        "--max_objects",
        type=int,
        default=None,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    p.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing/broken meshes instead of stopping.",
    )
    return p.parse_args(blender_cli_args())


def normalize_uid(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    return Path(value).stem


def read_object_list(path: Path):
    """
    Parse MinD-3D++ train/test object lists robustly.

    Supported examples:
        Christmas_tree 000-123/<uid>.glb
        Christmas_tree,000-123/<uid>.glb
        computer monitor,000-035/<uid>.glb

    Important:
    If a comma exists, parse CSV FIRST.  The previous implementation split on
    whitespace first, which can misparse a comma-separated line when the
    category itself contains spaces or when mixed formatting is present.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()

            if not line or line.startswith("#"):
                continue

            category = ""
            object_path = ""

            # MinD-3D++ list files may be comma-separated.  Always give CSV
            # parsing priority when a comma is present.
            if "," in line:
                csv_parts = next(csv.reader([line]))
                csv_parts = [x.strip() for x in csv_parts if x.strip()]

                if len(csv_parts) < 2:
                    raise ValueError(
                        f"{path}:{line_no}: expected category,object_path; "
                        f"got {line!r}"
                    )

                category = csv_parts[0].strip()
                object_path = csv_parts[-1].strip().replace("\\", "/")

            else:
                # Official whitespace-separated form:
                #     category   000-000/<uid>.glb
                parts = line.split()

                if len(parts) < 2:
                    raise ValueError(
                        f"{path}:{line_no}: expected category + object_path, "
                        f"got {line!r}"
                    )

                object_path = parts[-1].strip().replace("\\", "/")
                category = " ".join(parts[:-1]).strip()

            # Ignore a simple header.
            if line_no == 1:
                if (
                    "category" in category.lower()
                    and any(
                        key in object_path.lower()
                        for key in ("uid", "path", "object")
                    )
                ):
                    continue

            # Defensive cleanup: if a malformed path still contains a leading
            # "category," prefix, keep only the actual Objaverse path portion.
            if "," in object_path:
                object_path = object_path.split(",", 1)[-1].strip()

            uid = normalize_uid(object_path)

            if not uid:
                raise ValueError(f"{path}:{line_no}: empty UID")

            records.append(
                {
                    "category": category,
                    "object_path": object_path,
                    "uid": uid,
                }
            )

    return records


def clear_scene():
    """Delete all scene objects and unused world state."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove orphan collections/data as much as practical between objects.
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def setup_render_scene(resolution: int):
    """
    Headless-safe render setup.

    Eevee can produce completely black frames on some Linux/headless Blender
    installations because it depends on a working OpenGL context. Cycles is
    slower but much more reliable in `blender -b`.
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Keep the test render reasonably fast. The GT renders are evaluation
    # references rather than production-quality images.
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 16
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass

    print(
        f"[Blender] version={bpy.app.version_string}, "
        f"render_engine={scene.render.engine}"
    )

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    # RGB avoids any alpha-preview ambiguity in external file browsers.
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    # Rebuild the World shader explicitly so a custom/default startup file
    # cannot leave an unconnected/black world.
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")

    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    nodes.clear()

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Stable color transform for metric rendering.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "Medium High Contrast"
    except Exception:
        try:
            scene.view_settings.view_transform = "Standard"
        except Exception:
            pass

    return scene


def import_glb(mesh_path: Path):
    before = {obj.as_pointer() for obj in bpy.context.scene.objects}

    bpy.ops.import_scene.gltf(filepath=str(mesh_path))

    imported = [
        obj
        for obj in bpy.context.scene.objects
        if obj.as_pointer() not in before
    ]

    # Embedded cameras/lights from the GLB should not affect evaluation renders.
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    imported = [
        obj
        for obj in bpy.context.scene.objects
        if obj.as_pointer() not in before
    ]

    mesh_objects = [obj for obj in imported if obj.type == "MESH"]

    if not mesh_objects:
        raise ValueError(f"No mesh objects found in {mesh_path}")

    return imported, mesh_objects


def world_bbox_corners(mesh_objects):
    points = []

    for obj in mesh_objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))

    if not points:
        raise ValueError("No bounding-box points available")

    return points


def center_and_normalize(imported_objects, mesh_objects, target_extent: float):
    """
    Preserve internal hierarchy/materials and normalize the imported asset as a group.
    """
    corners = world_bbox_corners(mesh_objects)

    min_v = Vector((
        min(p.x for p in corners),
        min(p.y for p in corners),
        min(p.z for p in corners),
    ))
    max_v = Vector((
        max(p.x for p in corners),
        max(p.y for p in corners),
        max(p.z for p in corners),
    ))

    center = (min_v + max_v) * 0.5
    extent = max_v - min_v
    max_extent = max(extent.x, extent.y, extent.z)

    if max_extent <= 1e-8:
        raise ValueError("Degenerate object bounding box")

    scale = float(target_extent) / float(max_extent)

    imported_set = set(imported_objects)
    top_level = [
        obj
        for obj in imported_objects
        if obj.parent is None or obj.parent not in imported_set
    ]

    root = bpy.data.objects.new("__GT_NORMALIZE_ROOT__", None)
    bpy.context.scene.collection.objects.link(root)

    # Preserve each top-level object's world transform while reparenting.
    for obj in top_level:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world

    # p' = scale * (p - center)
    root.scale = (scale, scale, scale)
    root.location = (
        -scale * center.x,
        -scale * center.y,
        -scale * center.z,
    )

    bpy.context.view_layer.update()

    # Exact post-normalization bounding sphere radius from original bbox corners.
    normalized_points = [
        (p - center) * scale
        for p in corners
    ]
    sphere_radius = max(p.length for p in normalized_points)

    return {
        "original_bbox_min": list(min_v),
        "original_bbox_max": list(max_v),
        "original_center": list(center),
        "normalization_scale": scale,
        "normalized_sphere_radius": float(sphere_radius),
    }


def add_camera(fov_deg: float):
    data = bpy.data.cameras.new("GT_Camera")
    camera = bpy.data.objects.new("GT_Camera", data)
    bpy.context.scene.collection.objects.link(camera)

    data.type = "PERSP"
    data.angle = math.radians(float(fov_deg))
    data.lens_unit = "FOV"
    data.clip_start = 0.01
    data.clip_end = 1000.0

    bpy.context.scene.camera = camera
    return camera


def look_at(camera, target=Vector((0.0, 0.0, 0.0))):
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def camera_position(radius, azimuth_deg, elevation_deg):
    """
    Canonical convention:
      azimuth 0 deg -> camera on -Y axis, looking toward origin.
      positive azimuth rotates camera toward +X.
      elevation is measured from the XY plane.
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))

    cos_el = math.cos(el)

    x = radius * math.sin(az) * cos_el
    y = -radius * math.cos(az) * cos_el
    z = radius * math.sin(el)

    return Vector((x, y, z))


def get_world_points(mesh_objects):
    """
    Return the world-space corners of all object bounding boxes.

    Using the complete set of object bboxes is sufficient for conservative
    framing while avoiding an expensive walk through every mesh vertex.
    """
    points = []
    for obj in mesh_objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        raise ValueError("No bounding-box points available")
    return points


def projected_occupancy(scene, camera, points):
    """
    Return projected bbox occupancy in normalized image coordinates.

    IMPORTANT:
    `world_to_camera_view(...).z` is not used to decide whether a point is in
    front of the camera.  Its depth convention is not the same thing as the
    camera's local -Z forward convention and caused valid views to be rejected.

    Blender cameras look along local -Z, so a world point is in front iff its
    camera-local z coordinate is negative.
    """
    # Make sure matrix_world reflects the location/rotation just assigned.
    bpy.context.view_layer.update()

    cam_inv = camera.matrix_world.inverted()
    local_points = [cam_inv @ p for p in points]

    clip_start = max(float(camera.data.clip_start), 1e-6)

    # Camera forward is local -Z.  q.z >= -clip_start means at/behind near plane.
    if any(
        (not math.isfinite(q.x))
        or (not math.isfinite(q.y))
        or (not math.isfinite(q.z))
        or q.z >= -clip_start
        for q in local_points
    ):
        return float("inf"), None

    coords = [world_to_camera_view(scene, camera, p) for p in points]

    if any(
        (not math.isfinite(c.x))
        or (not math.isfinite(c.y))
        for c in coords
    ):
        return float("inf"), None

    xs = [float(c.x) for c in coords]
    ys = [float(c.y) for c in coords]

    stats = {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
        "min_camera_depth": float(min(-q.z for q in local_points)),
        "max_camera_depth": float(max(-q.z for q in local_points)),
    }
    stats["occupancy"] = max(stats["width"], stats["height"])

    return float(stats["occupancy"]), stats


def fit_shared_camera_radius(
    scene,
    camera,
    mesh_objects,
    base_radius,
    azimuth_offset_deg,
    target_fill,
):
    """
    Fit ONE camera radius shared by all six Zero123++ views.

    This is intentionally global rather than per-view:
      - prevents zoom changes between views,
      - preserves the six-view camera geometry,
      - guarantees every view fits inside the image,
      - is much more stable for PSNR/SSIM/LPIPS.
    """
    if not (0.05 < target_fill < 0.95):
        raise ValueError("--target_fill must be between 0.05 and 0.95")

    points = get_world_points(mesh_objects)

    def max_occ(radius):
        values = []
        details = []
        for azimuth, elevation in zip(ZP11_AZIMUTHS, ZP11_ELEVATIONS):
            azimuth_abs = azimuth + azimuth_offset_deg
            camera.location = camera_position(radius, azimuth_abs, elevation)
            look_at(camera)
            occ, stats = projected_occupancy(scene, camera, points)
            values.append(occ)
            details.append(stats)
        return max(values), details

    # Start from the conservative sphere-based radius and move outward until
    # all views are definitely valid and below the target occupancy.
    high = max(float(base_radius), 0.1)
    occ_high, _ = max_occ(high)

    expand_count = 0
    while (not math.isfinite(occ_high) or occ_high > target_fill) and expand_count < 20:
        high *= 1.35
        occ_high, _ = max_occ(high)
        expand_count += 1

    if not math.isfinite(occ_high):
        # This should be unreachable once camera-local depth is checked correctly.
        # Print geometry/camera diagnostics instead of silently producing bad views.
        max_point_radius = max(p.length for p in points)
        raise RuntimeError(
            "Could not find a valid camera distance for all six views even after "
            f"expansion. base_radius={base_radius:.6f}, "
            f"last_radius={high:.6f}, max_point_radius={max_point_radius:.6f}. "
            "Check the imported GLB transform/bounding box."
        )

    # Find a lower bound that is too close / too large.
    low = high / 2.0
    occ_low, _ = max_occ(low)
    shrink_count = 0
    while math.isfinite(occ_low) and occ_low < target_fill and shrink_count < 20:
        high = low
        low /= 1.35
        occ_low, _ = max_occ(low)
        shrink_count += 1

    # Monotonic binary search: farther camera -> smaller occupancy.
    for _ in range(40):
        mid = 0.5 * (low + high)
        occ_mid, _ = max_occ(mid)

        if (not math.isfinite(occ_mid)) or occ_mid > target_fill:
            low = mid
        else:
            high = mid

    radius = high
    max_final, per_view = max_occ(radius)

    return radius, max_final, per_view


def compute_camera_distance(sphere_radius, fov_deg, margin):
    """
    Distance that contains a sphere inside the square camera frustum.
    """
    half_fov = math.radians(float(fov_deg)) * 0.5
    if not (0.0 < half_fov < math.pi / 2):
        raise ValueError("--fov_deg must be in (0, 180)")

    # Angular-radius fit for a sphere:
    # sin(theta) = R / D  =>  D = R / sin(theta)
    distance = sphere_radius / max(math.sin(half_fov), 1e-6)
    return float(distance * margin)


def render_one_object(
    mesh_path: Path,
    output_dir: Path,
    record,
    args,
):
    # Reset the entire scene, not only objects. This prevents stale compositor,
    # World, or view-layer settings from producing black renders.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = setup_render_scene(args.resolution)

    imported, mesh_objects = import_glb(mesh_path)

    norm_meta = center_and_normalize(
        imported,
        mesh_objects,
        target_extent=args.object_extent,
    )

    # Metric-normalized illumination: use the uniform white world only.
    # No directional lights are added, which avoids view-dependent brightness.
    camera = add_camera(args.fov_deg)

    base_camera_radius = compute_camera_distance(
        sphere_radius=norm_meta["normalized_sphere_radius"],
        fov_deg=args.fov_deg,
        margin=args.camera_margin,
    )

    camera_radius, max_occ, fitted_stats = fit_shared_camera_radius(
        scene=scene,
        camera=camera,
        mesh_objects=mesh_objects,
        base_radius=base_camera_radius,
        azimuth_offset_deg=args.azimuth_offset_deg,
        target_fill=args.target_fill,
    )

    print(
        f"    bbox_min={norm_meta['original_bbox_min']} "
        f"bbox_max={norm_meta['original_bbox_max']} "
        f"scale={norm_meta['normalization_scale']:.6g} "
        f"sphere_R={norm_meta['normalized_sphere_radius']:.4f} "
        f"camera_R={camera_radius:.4f} "
        f"max_occ={max_occ:.3f}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = []

    for view_idx, (azimuth, elevation) in enumerate(
        zip(ZP11_AZIMUTHS, ZP11_ELEVATIONS)
    ):
        azimuth_abs = azimuth + args.azimuth_offset_deg

        # IMPORTANT: exactly the same radius for all six views.
        camera.location = camera_position(
            camera_radius,
            azimuth_abs,
            elevation,
        )
        look_at(camera)

        occ, frame_stats = projected_occupancy(
            scene,
            camera,
            get_world_points(mesh_objects),
        )

        out_path = output_dir / f"{view_idx:02d}.png"
        scene.render.filepath = str(out_path)

        bpy.ops.render.render(write_still=True)

        rr = bpy.data.images.get("Render Result")
        mean_rgb = None
        if rr is not None and len(rr.pixels) >= 4:
            pix = rr.pixels
            stride = max(4, (len(pix) // 4096 // 4) * 4)
            rgb_sum = 0.0
            rgb_count = 0
            for j in range(0, len(pix), stride):
                if j + 2 >= len(pix):
                    break
                rgb_sum += float(pix[j]) + float(pix[j + 1]) + float(pix[j + 2])
                rgb_count += 3
            mean_rgb = rgb_sum / max(rgb_count, 1)

            print(
                f"    view {view_idx:02d}: az={azimuth_abs:.1f}, "
                f"el={elevation:.1f}, occ={occ:.3f}, "
                f"mean_rgb={mean_rgb:.5f}"
            )

            if mean_rgb < 1e-4:
                raise RuntimeError(
                    "Render Result is essentially all black."
                )

        rendered.append(
            {
                "view_index": view_idx,
                "file": out_path.name,
                "zero123pp_relative_azimuth_deg": azimuth,
                "render_azimuth_deg": azimuth_abs,
                "elevation_deg": elevation,
                "camera_radius": float(camera_radius),
                "camera_location": [
                    float(camera.location.x),
                    float(camera.location.y),
                    float(camera.location.z),
                ],
                "frame_stats": frame_stats,
                "mean_rgb": None if mean_rgb is None else float(mean_rgb),
            }
        )

    meta = {
        "uid": record["uid"],
        "category": record["category"],
        "object_path": record["object_path"],
        "source_mesh": str(mesh_path),
        "zero123pp_version": "v1.1",
        "zero123pp_azimuths_relative_deg": ZP11_AZIMUTHS,
        "zero123pp_elevations_absolute_deg": ZP11_ELEVATIONS,
        "canonical_azimuth_definition": (
            "0 deg = camera at (0,-R,0), looking at origin; "
            "positive azimuth rotates toward +X"
        ),
        "azimuth_offset_deg": args.azimuth_offset_deg,
        "camera_type": "PERSP",
        "fov_deg": args.fov_deg,
        "fov_note": (
            "User-selected GT rendering FOV; Zero123++ v1.1 official README "
            "does not define a fixed output FOV."
        ),
        "resolution": [args.resolution, args.resolution],
        "object_extent": args.object_extent,
        "camera_margin": args.camera_margin,
        "camera_radius": camera_radius,
        "target_fill": args.target_fill,
        "framing": "single shared camera radius fitted across all six views",
        "lighting": "uniform white world only; no directional lights",
        "background": "white",
        "normalization": norm_meta,
        "views": rendered,
    }

    with (output_dir / "render_meta.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    gt_mesh_root = Path(args.gt_mesh_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    test_list = (
        Path(args.test_list).expanduser().resolve()
        if args.test_list
        else data_path / "test_list.txt"
    )

    records = read_object_list(test_list)

    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")

    records = records[args.start_index:]

    if args.max_objects is not None:
        records = records[:args.max_objects]

    print("=" * 80)
    print("[MinD-3D++ GT Render]")
    print("test_list    :", test_list)
    print("gt_mesh_root :", gt_mesh_root)
    print("output_dir   :", output_root)
    print("objects      :", len(records))
    print("resolution   :", args.resolution)
    print("FOV          :", args.fov_deg, "(user-selected; not official ZP++ v1.1)")
    print("azimuths     :", ZP11_AZIMUTHS)
    print("elevations   :", ZP11_ELEVATIONS)
    print("=" * 80)

    output_root.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for local_idx, record in enumerate(records):
        global_idx = args.start_index + local_idx

        mesh_path = gt_mesh_root / record["object_path"]
        object_out = output_root / record["uid"]

        expected = [
            object_out / f"{i:02d}.png"
            for i in range(6)
        ]

        if (
            not args.overwrite
            and all(p.is_file() for p in expected)
        ):
            print(
                f"[{global_idx:04d}] SKIP existing "
                f"{record['uid']} ({record['category']})"
            )
            skipped += 1
            continue

        if not mesh_path.is_file():
            msg = f"Missing mesh: {mesh_path}"
            if args.skip_missing:
                print(f"[{global_idx:04d}] WARNING {msg}")
                skipped += 1
                continue
            raise FileNotFoundError(msg)

        print(
            f"[{global_idx:04d}] Rendering "
            f"{record['category']} | {record['uid']}"
        )

        try:
            render_one_object(
                mesh_path=mesh_path,
                output_dir=object_out,
                record=record,
                args=args,
            )
            success += 1

        except Exception as exc:
            failed += 1
            print(
                f"[{global_idx:04d}] ERROR "
                f"{record['uid']}: {type(exc).__name__}: {exc}"
            )

            if not args.skip_missing:
                raise

    clear_scene()

    print()
    print("=" * 80)
    print("[DONE]")
    print("success :", success)
    print("skipped :", skipped)
    print("failed  :", failed)
    print("output  :", output_root)
    print("=" * 80)


if __name__ == "__main__":
    main()
