#!/usr/bin/env python3
"""
Render EEG-3D model3d assets using the exact 6 input camera poses expected by
InstantMesh / Zero123++.

This version includes:
1) Transparent RGBA rendering followed by EXACT pure-white compositing in Python.
2) Shared 3D normalization with alpha-silhouette-based scale refinement.
3) Optional neutral EMISSION material override for appearance/domain diagnostics.
4) Low-intensity studio lighting for original materials to avoid white-object saturation.
5) Correct bottom-origin 3x2 grid assembly (no per-view vertical flip).
6) Lighting is instantiated exactly once per asset.
7) Optional training export: fixed 6 input + novel target views with RGB/alpha/depth/world-normal/cameras.

Run with Blender:
    blender -b -P render_eeg3d_instantmesh.py -- \
        --input /path/to/model3d/xxx.glb \
        --output /path/to/output

Batch directory:
    blender -b -P render_eeg3d_instantmesh.py -- \
        --input /path/to/model3d \
        --output /path/to/rendered \
        --recursive

Outputs per asset:
    <output>/<asset_name>/
        00.png ... 05.png               # final exact-white composited views
        instantmesh_grid.png            # 640 x 960, 3 rows x 2 cols
        instantmesh_cameras.npz         # c2ws [6,4,4], Ks [6,3,3]
        instantmesh_cameras_16d.npy     # InstantMesh encoder format [6,16]
        metadata.json

Optional debug outputs:
        rgba_raw/00.png ... 05.png      # final raw transparent renders (post-scale-refinement)
        alpha_probe/                    # temporary probe renders if --keep_probes is set

Camera convention matches TencentARC/InstantMesh:
    azimuths   = [30, 90, 150, 210, 270, 330] deg
    elevations = [20,-10, 20,-10, 20,-10] deg
    radius     = 4.0
    FOV        = 30 deg

IMPORTANT:
- The rendered images already have the correct camera intrinsics.
- Do NOT independently crop, recenter, or resize each view afterward.
- Feed instantmesh_grid.png directly to the original InstantMesh 3x2 splitter.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


AZIMUTHS_DEG = np.array([30, 90, 150, 210, 270, 330], dtype=np.float64)
ELEVATIONS_DEG = np.array([20, -10, 20, -10, 20, -10], dtype=np.float64)
SUPPORTED_EXTS = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".blend"}

SCRIPT_VERSION = "2026-08-19-v4-train-export-rgb-alpha-depth-normal"



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="A model file or a directory containing EEG-3D model3d assets.")
    p.add_argument("--output", required=True, help="Output root directory.")
    p.add_argument("--recursive", action="store_true",
                   help="Recursively search the input directory.")
    p.add_argument("--resolution", type=int, default=320,
                   help="Per-view resolution. InstantMesh default is 320.")
    p.add_argument("--radius", type=float, default=4.0,
                   help="Camera radius. Keep 4.0 for exact InstantMesh input cameras.")
    p.add_argument("--fov", type=float, default=30.0,
                   help="Camera FOV in degrees. Keep 30 for InstantMesh.")
    p.add_argument("--fill", type=float, default=0.85,
                   help="Target maximum silhouette extent in the image (0-1).")
    p.add_argument("--engine", choices=["eevee", "cycles"], default="eevee")
    p.add_argument("--samples", type=int, default=64,
                   help="Render samples (mainly relevant to Cycles).")
    p.add_argument("--render_exposure", type=float, default=0.0,
                   help="Blender render exposure for object appearance. Default 0.0 keeps pure white truly 255.")
    p.add_argument("--no_smooth", action="store_true",
                   help="Do not enable smooth shading on imported meshes.")
    p.add_argument("--save_rgba", action="store_true",
                   help="Also save intermediate transparent RGBA renders under rgba_raw/.")
    p.add_argument("--keep_probes", action="store_true",
                   help="Keep alpha probe renders used for scale refinement.")
    p.add_argument("--max_vertices", type=int, default=200000,
                   help="Maximum number of projected vertices used for coarse scale initialization.")
    p.add_argument("--probe_resolution", type=int, default=128,
                   help="Resolution used for alpha silhouette scale refinement.")
    p.add_argument("--auto_scale_iters", type=int, default=3,
                   help="Number of alpha silhouette refinement iterations.")
    p.add_argument("--alpha_threshold", type=float, default=0.02,
                   help="Alpha threshold used when measuring silhouette extent.")
    p.add_argument("--material_override", action="store_true",
                   help="Override all materials with a neutral unlit emission material for geometry/camera diagnostics.")
    p.add_argument("--override_color", type=float, nargs=3, default=(0.65, 0.65, 0.65),
                   metavar=("R", "G", "B"),
                   help="RGB color for neutral material override, range [0,1].")
    p.add_argument("--override_roughness", type=float, default=0.7,
                   help="Deprecated/ignored in emission override mode; kept for CLI compatibility.")
    p.add_argument("--world_strength", type=float, default=0.15,
                   help="World illumination strength in normal-material mode. Default 0.15.")
    p.add_argument("--key_energy", type=float, default=500.0,
                   help="Key area-light energy in normal-material mode.")
    p.add_argument("--fill_energy", type=float, default=150.0,
                   help="Fill area-light energy in normal-material mode.")
    p.add_argument("--rim_energy", type=float, default=250.0,
                   help="Rim area-light energy in normal-material mode.")
    p.add_argument("--sun_energy", type=float, default=0.25,
                   help="Sun-light energy in normal-material mode.")

    # Training dataset export. The first six views always remain the exact
    # InstantMesh / Zero123++ input cameras; additional views are novel-view
    # supervision targets.
    p.add_argument("--training_export", action="store_true",
                   help="Also export RGB/alpha/depth/normal/camera data for InstantMesh fine-tuning.")
    p.add_argument("--target_views", type=int, default=26,
                   help="Number of additional target views. Default 26 gives 32 total views (6 input + 26 target).")
    p.add_argument("--target_elevation_min", type=float, default=-30.0,
                   help="Minimum target-view elevation in degrees.")
    p.add_argument("--target_elevation_max", type=float, default=40.0,
                   help="Maximum target-view elevation in degrees.")
    p.add_argument("--target_azimuth_offset", type=float, default=17.0,
                   help="Azimuth offset for deterministic low-discrepancy target views.")
    p.add_argument("--depth_scale", type=float, default=6.0,
                   help="Depth normalization scale. 6.0 matches the official InstantMesh Objaverse loader.")
    p.add_argument("--scale_probe_target_views", type=int, default=6,
                   help="How many target cameras to include during shared scale refinement.")
    p.add_argument("--save_training_previews", action="store_true",
                   help="Save human-readable alpha/depth/normal PNG previews in addition to authoritative .npy arrays.")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Scene / import helpers
# -----------------------------------------------------------------------------

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights, bpy.data.images, bpy.data.materials):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def import_obj(path):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))


def import_ply(path):
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))


def import_stl(path):
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))


def import_blend_objects(path):
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    for obj in data_to.objects:
        if obj is not None:
            try:
                bpy.context.scene.collection.objects.link(obj)
            except RuntimeError:
                pass
    return [o for o in bpy.data.objects if o not in before]


def import_model(path: Path):
    before = set(bpy.data.objects)
    ext = path.suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        import_obj(path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif ext == ".ply":
        import_ply(path)
    elif ext == ".stl":
        import_stl(path)
    elif ext == ".blend":
        return import_blend_objects(path)
    else:
        raise ValueError(f"Unsupported model format: {path}")
    return [o for o in bpy.data.objects if o not in before]


def remove_imported_cameras_and_lights():
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def mesh_objects():
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def set_smooth_shading(objects):
    for obj in objects:
        if obj.type != "MESH":
            continue
        for poly in obj.data.polygons:
            poly.use_smooth = True


def apply_material_override(meshes, color=(0.55, 0.55, 0.55), roughness=0.7):
    """Apply an UNLIT emission material for geometry/camera diagnostics.

    The previous Principled-BSDF override was still affected by studio/world
    illumination and could saturate toward white.  This emission-only material
    produces a nearly constant RGB value independent of view direction or lights,
    so failure in this mode is much more indicative of geometry/camera fusion.

    `roughness` is accepted only for backward CLI compatibility and is ignored.
    """
    del roughness
    color = tuple(float(np.clip(c, 0.0, 1.0)) for c in color)

    mat = bpy.data.materials.new(name="InstantMesh_NeutralEmissionOverride")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    # ShaderNodeEmission is supported by the Blender versions targeted here.
    # Fall back to Principled emission inputs if a build removes the node.
    try:
        emission = nt.nodes.new("ShaderNodeEmission")
        emission.location = (0, 0)
        emission.inputs["Color"].default_value = (color[0], color[1], color[2], 1.0)
        emission.inputs["Strength"].default_value = 1.0
        nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    except Exception:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        try:
            bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        except Exception:
            pass
        emission_key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
        if emission_key in bsdf.inputs:
            bsdf.inputs[emission_key].default_value = (color[0], color[1], color[2], 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.0
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 1.0
        nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    try:
        mat.diffuse_color = (color[0], color[1], color[2], 1.0)
    except Exception:
        pass

    for obj in meshes:
        if obj.type != "MESH":
            continue
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat
        try:
            obj.active_material = mat
        except Exception:
            pass


def world_mesh_vertices(objects, max_vertices=200000):
    """Collect world-space mesh vertices. Subsample if very large."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        try:
            if mesh is None or len(mesh.vertices) == 0:
                continue
            mat = obj_eval.matrix_world
            arr = np.empty((len(mesh.vertices), 3), dtype=np.float64)
            for i, v in enumerate(mesh.vertices):
                p = mat @ v.co
                arr[i] = (p.x, p.y, p.z)
            chunks.append(arr)
        finally:
            obj_eval.to_mesh_clear()

    if not chunks:
        raise RuntimeError("No mesh vertices found after import.")

    verts = np.concatenate(chunks, axis=0)
    if verts.shape[0] > max_vertices:
        idx = np.linspace(0, verts.shape[0] - 1, max_vertices, dtype=np.int64)
        verts = verts[idx]
    return verts


# -----------------------------------------------------------------------------
# Camera helpers
# -----------------------------------------------------------------------------

def center_looking_at_c2w(camera_position):
    """
    Exact InstantMesh/OpenGL convention:
      camera looks along local -Z, local +Y is up, local +X is right
      world up is +Z
    Blender cameras use the same local convention, so matrix_world = c2w.
    """
    pos = np.asarray(camera_position, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    z_axis = pos.copy()
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = pos
    return c2w


def get_instantmesh_c2ws(radius):
    az = np.deg2rad(AZIMUTHS_DEG)
    el = np.deg2rad(ELEVATIONS_DEG)
    xs = radius * np.cos(el) * np.cos(az)
    ys = radius * np.cos(el) * np.sin(az)
    zs = radius * np.sin(el)
    positions = np.stack([xs, ys, zs], axis=-1)
    return np.stack([center_looking_at_c2w(p) for p in positions], axis=0)



def get_training_target_cameras(num_views, radius, elevation_min=-30.0,
                                elevation_max=40.0, azimuth_offset=17.0):
    """Generate deterministic low-discrepancy target cameras on a spherical band.

    The six InstantMesh input cameras are NOT generated here.  These are only
    additional novel-view supervision cameras.  Elevation is stratified uniformly
    in sin(elevation), while azimuth follows a golden-angle sequence.
    """
    num_views = int(num_views)
    if num_views <= 0:
        return np.empty((0, 4, 4), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    el_min = math.radians(float(elevation_min))
    el_max = math.radians(float(elevation_max))
    if el_min >= el_max:
        raise ValueError("target_elevation_min must be smaller than target_elevation_max")

    sin_min = math.sin(el_min)
    sin_max = math.sin(el_max)
    golden_angle = 137.50776405003785

    azimuths = []
    elevations = []
    c2ws = []
    for i in range(num_views):
        u = (i + 0.5) / num_views
        sin_el = sin_min + u * (sin_max - sin_min)
        el = math.asin(float(np.clip(sin_el, -1.0, 1.0)))
        az_deg = (float(azimuth_offset) + i * golden_angle) % 360.0
        az = math.radians(az_deg)

        pos = np.array([
            radius * math.cos(el) * math.cos(az),
            radius * math.cos(el) * math.sin(az),
            radius * math.sin(el),
        ], dtype=np.float64)
        c2ws.append(center_looking_at_c2w(pos))
        azimuths.append(az_deg)
        elevations.append(math.degrees(el))

    return (
        np.stack(c2ws, axis=0),
        np.asarray(azimuths, dtype=np.float64),
        np.asarray(elevations, dtype=np.float64),
    )


def choose_scale_probe_cameras(input_c2ws, target_c2ws, max_target_views=6):
    """Use all six input cameras plus a small evenly spaced target subset for scaling."""
    if target_c2ws.shape[0] == 0 or max_target_views <= 0:
        return input_c2ws
    n = min(int(max_target_views), target_c2ws.shape[0])
    idx = np.linspace(0, target_c2ws.shape[0] - 1, n, dtype=np.int64)
    return np.concatenate([input_c2ws, target_c2ws[idx]], axis=0)


def get_normalized_K(fov_deg):
    f = 0.5 / math.tan(math.radians(fov_deg) * 0.5)
    return np.array(
        [[f, 0.0, 0.5],
         [0.0, f, 0.5],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_camera(c2w, fov_deg):
    cam_data = bpy.data.cameras.new("InstantMeshCamera")
    cam = bpy.data.objects.new("InstantMeshCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    cam.data.type = "PERSP"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = 0.5 * cam.data.sensor_width / math.tan(math.radians(fov_deg) * 0.5)
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0
    cam.data.clip_start = 0.05
    cam.data.clip_end = 100.0
    cam.matrix_world = Matrix(c2w.tolist())
    return cam


# -----------------------------------------------------------------------------
# Lighting / render config
# -----------------------------------------------------------------------------

def aim_object_at(obj, target=(0.0, 0.0, 0.0)):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(name, location, energy, size):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = Vector(location)
    aim_object_at(obj)
    return obj


def add_sun_light(name, location, energy, angle_deg=8.0):
    data = bpy.data.lights.new(name=name, type="SUN")
    data.energy = energy
    try:
        data.angle = math.radians(angle_deg)
    except Exception:
        pass
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = Vector(location)
    aim_object_at(obj)
    return obj


def setup_lighting(key_energy=500.0, fill_energy=150.0,
                   rim_energy=250.0, sun_energy=0.25):
    """Low-intensity studio lighting for original-material renders.

    Energies are deliberately much lower than the previous 2200/900/1400/1.2
    configuration, which strongly saturated white EEG-3D assets against the
    white InstantMesh background.
    """
    add_area_light("Key",  (5.5, -4.0, 5.5), float(key_energy), 5.0)
    add_area_light("Fill", (-4.0, -3.0, 3.5), float(fill_energy), 5.0)
    add_area_light("Rim",  (0.0,  5.0, 5.0), float(rim_energy), 4.5)
    add_sun_light("Sun", (3.0, -3.0, 6.0), float(sun_energy), 10.0)


# keep backward compatibility for older call sites
setup_studio_lighting = setup_lighting


def setup_world_for_rgba(world_strength=0.15):
    """Configure transparent film while controlling WORLD illumination only.

    The world remains invisible because film_transparent=True.  Its strength still
    affects PBR surface lighting, so keeping it low is important for white objects.
    In emission-override diagnostic mode use world_strength=0.0.
    """
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs["Strength"].default_value = float(world_strength)
    bpy.context.scene.render.film_transparent = True


def setup_render(resolution, engine, samples, exposure=0.0):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    scene.view_settings.exposure = float(exposure)
    scene.view_settings.gamma = 1.0

    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            scene.render.engine = "BLENDER_EEVEE"
        try:
            scene.eevee.use_gtao = True
            scene.eevee.gtao_factor = 1.0
        except Exception:
            pass


def render_view_rgba(cam, c2w, output_path):
    cam.matrix_world = Matrix(c2w.tolist())
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


# -----------------------------------------------------------------------------
# Training supervision passes
# -----------------------------------------------------------------------------

def render_view_to_array(cam, c2w, temp_path):
    """Render current scene/material setup to disk and reload as RGBA float32.

    Blender 4.5 background rendering can expose a zero-sized in-memory
    ``Render Result`` to Python in some configurations.  The previous training
    exporter read that image directly, which could propagate an empty [0,0,4]
    array and later fail at ``Image.pixels`` assignment.

    Writing a real 16-bit RGBA PNG first and loading it back is much more robust
    in headless Blender and still gives sufficient precision for our encoded
    normal/depth supervision passes.
    """
    temp_path = Path(temp_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    cam.matrix_world = Matrix(c2w.tolist())
    bpy.context.view_layer.update()

    prev_filepath = scene.render.filepath
    prev_format = scene.render.image_settings.file_format
    prev_mode = scene.render.image_settings.color_mode
    prev_depth = scene.render.image_settings.color_depth

    try:
        scene.render.filepath = str(temp_path)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "16"
        bpy.ops.render.render(write_still=True)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Training render was not written: {temp_path}")

        image = bpy.data.images.load(str(temp_path), check_existing=False)
        try:
            w, h = image.size[:]
            if w <= 0 or h <= 0:
                raise RuntimeError(
                    f"Loaded training render has invalid size {w}x{h}: {temp_path}"
                )
            flat = np.asarray(image.pixels[:], dtype=np.float32)
            expected = int(w) * int(h) * 4
            if flat.size != expected:
                raise RuntimeError(
                    f"Loaded training render pixel count mismatch for {temp_path}: "
                    f"got {flat.size}, expected {expected} ({w}x{h}x4)"
                )
            return flat.reshape(h, w, 4).copy()
        finally:
            bpy.data.images.remove(image)
    finally:
        scene.render.filepath = prev_filepath
        scene.render.image_settings.file_format = prev_format
        scene.render.image_settings.color_mode = prev_mode
        scene.render.image_settings.color_depth = prev_depth


def create_world_normal_material():
    """Unlit material encoding WORLD-space normals from [-1,1] to [0,1]."""
    mat = bpy.data.materials.get("__InstantMesh_WorldNormal__")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("__InstantMesh_WorldNormal__")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    mul = nt.nodes.new("ShaderNodeVectorMath")
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = 'ADD'
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0

    nt.links.new(geom.outputs["Normal"], mul.inputs[0])
    nt.links.new(mul.outputs["Vector"], add.inputs[0])
    nt.links.new(add.outputs["Vector"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def create_camera_depth_material(depth_scale):
    """Unlit material encoding positive camera-space -Z / depth_scale as grayscale.

    This matches InstantMesh's positive mesh-render depth convention more closely
    than Euclidean camera range.  Values are clamped to [0,1] only for the render
    pass; the saved authoritative depth is decoded back by multiplying depth_scale.
    """
    name = f"__InstantMesh_CameraDepth_{float(depth_scale):.4f}__"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    transform = nt.nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = 'POINT'
    transform.convert_from = 'WORLD'
    transform.convert_to = 'CAMERA'
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    neg = nt.nodes.new("ShaderNodeMath")
    neg.operation = 'MULTIPLY'
    neg.inputs[1].default_value = -1.0
    div = nt.nodes.new("ShaderNodeMath")
    div.operation = 'DIVIDE'
    div.inputs[1].default_value = float(depth_scale)
    vmax = nt.nodes.new("ShaderNodeMath")
    vmax.operation = 'MAXIMUM'
    vmax.inputs[1].default_value = 0.0
    vmin = nt.nodes.new("ShaderNodeMath")
    vmin.operation = 'MINIMUM'
    vmin.inputs[1].default_value = 1.0
    combine = nt.nodes.new("ShaderNodeCombineXYZ")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0

    nt.links.new(geom.outputs["Position"], transform.inputs["Vector"])
    nt.links.new(transform.outputs["Vector"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], neg.inputs[0])
    nt.links.new(neg.outputs[0], div.inputs[0])
    nt.links.new(div.outputs[0], vmax.inputs[0])
    nt.links.new(vmax.outputs[0], vmin.inputs[0])
    nt.links.new(vmin.outputs[0], combine.inputs["X"])
    nt.links.new(vmin.outputs[0], combine.inputs["Y"])
    nt.links.new(vmin.outputs[0], combine.inputs["Z"])
    nt.links.new(combine.outputs["Vector"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def blender_to_top_left(array):
    """Convert Blender pixel-array bottom-origin convention to normal top-left arrays."""
    return np.flipud(array).copy()


def render_training_modalities(cam, c2w, depth_scale, temp_dir, view_tag, alpha_threshold=0.0):
    """Return RGB(A), alpha, camera-Z depth, and world normals for one camera.

    Arrays returned here use top-left image convention:
      rgb       [H,W,3] in [0,1], white-composited
      alpha     [H,W,1] in [0,1]
      depth     [H,W,1] positive camera-space depth, background 0
      normal    [H,W,3] encoded from world normal [-1,1] to [0,1], background 0
    """
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    previous_override = view_layer.material_override
    previous_transform = getattr(scene.view_settings, "view_transform", None)
    previous_exposure = scene.view_settings.exposure
    previous_gamma = scene.view_settings.gamma

    try:
        # Appearance render: use actual object materials (or CLI diagnostic material
        # if the caller deliberately replaced them before arriving here).
        view_layer.material_override = None
        rgba_bottom = render_view_to_array(cam, c2w, Path(temp_dir) / f"{view_tag}_rgb.png")
        alpha_bottom = np.clip(rgba_bottom[..., 3:4], 0.0, 1.0)
        rgb_white_bottom = composite_rgba_to_exact_white(rgba_bottom)[..., :3]

        # Numeric passes should bypass display transforms as much as Blender allows.
        try:
            scene.view_settings.view_transform = "Raw"
        except Exception:
            try:
                scene.view_settings.view_transform = "Standard"
            except Exception:
                pass
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

        view_layer.material_override = create_world_normal_material()
        normal_rgba_bottom = render_view_to_array(cam, c2w, Path(temp_dir) / f"{view_tag}_normal.png")
        normal_bottom = np.clip(normal_rgba_bottom[..., :3], 0.0, 1.0)

        view_layer.material_override = create_camera_depth_material(depth_scale)
        depth_rgba_bottom = render_view_to_array(cam, c2w, Path(temp_dir) / f"{view_tag}_depth.png")
        depth_bottom = np.clip(depth_rgba_bottom[..., 0:1], 0.0, 1.0) * float(depth_scale)

        # Use appearance alpha as the authoritative visibility mask. This prevents
        # material-override passes from supervising geometry that is fully hidden
        # by alpha/transparent source materials.
        visible = alpha_bottom
        if alpha_threshold > 0:
            visible = np.where(visible > alpha_threshold, visible, 0.0).astype(np.float32)
        depth_bottom = depth_bottom * visible
        normal_bottom = normal_bottom * visible

        return {
            "rgb": blender_to_top_left(rgb_white_bottom).astype(np.float32),
            "alpha": blender_to_top_left(alpha_bottom).astype(np.float32),
            "depth": blender_to_top_left(depth_bottom).astype(np.float32),
            "normal": blender_to_top_left(normal_bottom).astype(np.float32),
            # Keep bottom-origin RGB for bpy image saving without an extra flip.
            "rgb_bottom": rgb_white_bottom.astype(np.float32),
        }
    finally:
        view_layer.material_override = previous_override
        if previous_transform is not None:
            try:
                scene.view_settings.view_transform = previous_transform
            except Exception:
                pass
        scene.view_settings.exposure = previous_exposure
        scene.view_settings.gamma = previous_gamma


def save_training_preview_rgba(path, top_left_rgb, alpha=None):
    """Save a top-left RGB array through Blender by converting back to bottom-origin."""
    rgb = np.clip(top_left_rgb, 0.0, 1.0)
    if alpha is None:
        a = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
    else:
        a = np.clip(alpha, 0.0, 1.0)
    rgba_top = np.concatenate([rgb, a], axis=-1)
    save_image_pixels_rgba(path, np.flipud(rgba_top).copy())


def export_training_dataset(asset_dir, cam, input_c2ws, target_c2ws,
                            target_azimuths, target_elevations, args):
    """Export training supervision while preserving exact six fixed input cameras.

    Authoritative numeric supervision is saved as float32 .npy to avoid PNG color
    management/quantization ambiguity. RGB is saved as PNG because it is intended
    as an appearance input. Camera NPZ also includes `cam_poses` (w2c first 3 rows)
    for easy adaptation to the official ObjaverseData naming convention.
    """
    train_dir = asset_dir / "training"
    rgb_dir = train_dir / "rgb"
    alpha_dir = train_dir / "alpha"
    depth_dir = train_dir / "depth"
    normal_dir = train_dir / "normal"
    temp_dir = train_dir / ".render_tmp"
    for d in (rgb_dir, alpha_dir, depth_dir, normal_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    preview_alpha_dir = train_dir / "preview_alpha"
    preview_depth_dir = train_dir / "preview_depth"
    preview_normal_dir = train_dir / "preview_normal"
    if args.save_training_previews:
        for d in (preview_alpha_dir, preview_depth_dir, preview_normal_dir):
            d.mkdir(parents=True, exist_ok=True)

    all_c2ws = np.concatenate([input_c2ws, target_c2ws], axis=0)
    all_w2cs = np.linalg.inv(all_c2ws)
    K = get_normalized_K(args.fov).astype(np.float32)
    all_Ks = np.repeat(K[None], all_c2ws.shape[0], axis=0)
    input_indices = np.arange(input_c2ws.shape[0], dtype=np.int64)
    target_indices = np.arange(input_c2ws.shape[0], all_c2ws.shape[0], dtype=np.int64)

    max_depth = 0.0
    fills = []
    for idx, c2w in enumerate(all_c2ws):
        kind = "input" if idx < input_c2ws.shape[0] else "target"
        print(f"  [training] {idx:03d}/{all_c2ws.shape[0]-1:03d} ({kind})")
        data = render_training_modalities(
            cam, c2w,
            depth_scale=args.depth_scale,
            temp_dir=temp_dir,
            view_tag=f"{idx:03d}",
            alpha_threshold=0.0,
        )

        # RGB PNG: opaque pure-white background, same appearance distribution used
        # for inference.  save_image_pixels_rgba wants Blender-bottom-origin arrays.
        rgba_bottom = np.concatenate([
            data["rgb_bottom"],
            np.ones((*data["rgb_bottom"].shape[:2], 1), dtype=np.float32)
        ], axis=-1)
        save_image_pixels_rgba(rgb_dir / f"{idx:03d}.png", rgba_bottom)

        np.save(alpha_dir / f"{idx:03d}.npy", data["alpha"].astype(np.float32))
        np.save(depth_dir / f"{idx:03d}.npy", data["depth"].astype(np.float32))
        np.save(normal_dir / f"{idx:03d}.npy", data["normal"].astype(np.float32))

        if np.any(data["alpha"] > 0):
            max_depth = max(max_depth, float(data["depth"].max()))
            ys, xs = np.where(data["alpha"][..., 0] > args.alpha_threshold)
            if len(xs) > 0:
                fill = max(
                    (xs.max() - xs.min() + 1) / data["alpha"].shape[1],
                    (ys.max() - ys.min() + 1) / data["alpha"].shape[0],
                )
            else:
                fill = 0.0
        else:
            fill = 0.0
        fills.append(float(fill))

        if args.save_training_previews:
            alpha_rgb = np.repeat(data["alpha"], 3, axis=-1)
            save_training_preview_rgba(preview_alpha_dir / f"{idx:03d}.png", alpha_rgb)

            depth_preview = np.repeat(np.clip(data["depth"] / args.depth_scale, 0, 1), 3, axis=-1)
            save_training_preview_rgba(preview_depth_dir / f"{idx:03d}.png", depth_preview)

            save_training_preview_rgba(preview_normal_dir / f"{idx:03d}.png", data["normal"])

    # Official ObjaverseData stores w2c as [N,3,4] under `cam_poses`, then inverts
    # it to obtain c2w. Save that field as well as explicit matrices.
    np.savez(
        train_dir / "cameras.npz",
        cam_poses=all_w2cs[:, :3, :4].astype(np.float32),
        c2ws=all_c2ws.astype(np.float32),
        w2cs=all_w2cs.astype(np.float32),
        Ks=all_Ks.astype(np.float32),
        input_indices=input_indices,
        target_indices=target_indices,
        input_c2ws=input_c2ws.astype(np.float32),
        target_c2ws=target_c2ws.astype(np.float32),
        input_Ks=all_Ks[input_indices].astype(np.float32),
        target_Ks=all_Ks[target_indices].astype(np.float32),
    )

    fixed_az = AZIMUTHS_DEG.tolist()
    fixed_el = ELEVATIONS_DEG.tolist()
    manifest = {
        "format_version": 1,
        "authoritative_arrays": {
            "rgb": "training/rgb/%03d.png (RGB [0,1] after loading; exact white background)",
            "alpha": "training/alpha/%03d.npy float32 [H,W,1]",
            "depth": "training/depth/%03d.npy float32 [H,W,1], positive camera-space -Z, background=0",
            "normal": "training/normal/%03d.npy float32 [H,W,3], WORLD normal encoded to [0,1], background=0",
        },
        "input_indices": input_indices.tolist(),
        "target_indices": target_indices.tolist(),
        "num_input_views": int(len(input_indices)),
        "num_target_views": int(len(target_indices)),
        "num_total_views": int(all_c2ws.shape[0]),
        "camera": {
            "radius": float(args.radius),
            "fov_deg": float(args.fov),
            "normalized_K": K.tolist(),
            "input_azimuths_deg": fixed_az,
            "input_elevations_deg": fixed_el,
            "target_azimuths_deg": target_azimuths.tolist(),
            "target_elevations_deg": target_elevations.tolist(),
            "convention": "c2w OpenGL/Blender: local +X right, +Y up, camera looks -Z; world up +Z",
        },
        "depth": {
            "depth_scale": float(args.depth_scale),
            "observed_max_depth": float(max_depth),
            "warning": (
                "observed max reached depth_scale; increase --depth_scale"
                if max_depth >= args.depth_scale * 0.999 else None
            ),
        },
        "per_view_silhouette_fill": fills,
        "notes": [
            "The first six views are fixed InstantMesh/Zero123++ cameras and should be used as input views.",
            "Additional views are novel-view supervision targets.",
            "Do not crop/recenter/resize views independently; doing so changes effective intrinsics.",
            "Normals are world-space so a future camera/world rotation augmentation must rotate normals consistently.",
        ],
    }
    with open(train_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"  Training export saved: {train_dir}")
    print(f"  Views: {len(input_indices)} input + {len(target_indices)} target = {all_c2ws.shape[0]}")
    print(f"  Observed max depth: {max_depth:.4f} (depth_scale={args.depth_scale:.4f})")
    return manifest


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------

def load_image_pixels_rgba(path):
    img = bpy.data.images.load(str(path), check_existing=False)
    try:
        w, h = img.size[:]
        pixels = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        return pixels.copy()
    finally:
        bpy.data.images.remove(img)


def save_image_pixels_rgba(path, rgba):
    """Save a Blender-bottom-origin RGBA array with explicit shape validation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise RuntimeError(
            f"Invalid RGBA array for {path}: shape={rgba.shape}; expected [H,W,4]"
        )

    h, w, _ = rgba.shape
    if h <= 0 or w <= 0:
        raise RuntimeError(
            f"Refusing to save empty image {path}: shape={rgba.shape}. "
            "This usually means the preceding Blender render returned 0x0 pixels."
        )

    flat = np.clip(rgba, 0.0, 1.0).reshape(-1).astype(np.float32, copy=False)
    expected = int(h) * int(w) * 4
    if flat.size != expected:
        raise RuntimeError(
            f"RGBA pixel count mismatch for {path}: got {flat.size}, expected {expected}"
        )

    img = bpy.data.images.new(path.stem, width=int(w), height=int(h), alpha=True, float_buffer=False)
    try:
        # foreach_set is stricter and more stable than assigning to img.pixels directly.
        img.pixels.foreach_set(flat)
        img.update()
        img.filepath_raw = str(path)
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


def composite_rgba_to_exact_white(rgba):
    a = np.clip(rgba[..., 3:4], 0.0, 1.0)
    rgb = np.clip(rgba[..., :3], 0.0, 1.0)
    out_rgb = rgb * a + (1.0 - a)  # exact white bg
    out = np.concatenate([out_rgb, np.ones_like(a)], axis=-1)
    return out


def alpha_fill_ratio(rgba, alpha_threshold=0.02):
    alpha = rgba[..., 3]
    mask = alpha > alpha_threshold
    h, w = mask.shape
    if not np.any(mask):
        return 0.0
    ys, xs = np.where(mask)
    fill_x = (xs.max() - xs.min() + 1) / float(w)
    fill_y = (ys.max() - ys.min() + 1) / float(h)
    return float(max(fill_x, fill_y))


def save_grid_from_view_arrays(view_arrays, grid_path, resolution):
    """Compose exact InstantMesh layout, preserving Blender's bottom-origin pixel convention.

    The PNG must appear visually as:
        0 | 1
        2 | 3
        4 | 5

    `bpy.data.images[*].pixels` is bottom-origin. Therefore logical top-row views
    (0, 1) must be written into the highest Y block of the backing array. Do NOT
    flip the completed grid: that would also vertically flip every individual view.
    """
    H = W = resolution
    grid = np.ones((3 * H, 2 * W, 4), dtype=np.float32)

    for idx, rgba in enumerate(view_arrays):
        if rgba.shape[0] != H or rgba.shape[1] != W:
            raise RuntimeError(
                f"Unexpected rendered size for view {idx}: "
                f"{rgba.shape[:2]} != {(H, W)}"
            )

        logical_row = idx // 2  # 0=top, 1=middle, 2=bottom in the saved PNG
        col = idx % 2

        # Backing image array is bottom-origin.
        y0 = (2 - logical_row) * H
        x0 = col * W
        grid[y0:y0 + H, x0:x0 + W, :] = rgba

    save_image_pixels_rgba(grid_path, grid)


# -----------------------------------------------------------------------------
# Normalization / scaling helpers
# -----------------------------------------------------------------------------

def projected_fill(points_centered, scale, c2ws, fov_deg):
    """Coarse max projected extent proxy using projected mesh vertices."""
    points = points_centered * scale
    tan_half = math.tan(math.radians(fov_deg) * 0.5)
    worst = 0.0
    for c2w in c2ws:
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        p_cam = (points - t[None, :]) @ R
        z_forward = -p_cam[:, 2]
        valid = z_forward > 1e-6
        if not np.any(valid):
            return float("inf")
        p_cam = p_cam[valid]
        z_forward = z_forward[valid]
        x_ndc = (p_cam[:, 0] / z_forward) / tan_half
        y_ndc = (p_cam[:, 1] / z_forward) / tan_half
        fill_x = (x_ndc.max() - x_ndc.min()) * 0.5
        fill_y = (y_ndc.max() - y_ndc.min()) * 0.5
        worst = max(worst, float(fill_x), float(fill_y))
    return worst


def solve_uniform_scale(points_world, c2ws, fov_deg, target_fill):
    pmin = points_world.min(axis=0)
    pmax = points_world.max(axis=0)
    center = (pmin + pmax) * 0.5
    centered = points_world - center[None, :]

    lo = 0.0
    hi = 1.0
    for _ in range(40):
        f = projected_fill(centered, hi, c2ws, fov_deg)
        if not np.isfinite(f) or f >= target_fill:
            break
        hi *= 1.5
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        f = projected_fill(centered, mid, c2ws, fov_deg)
        if np.isfinite(f) and f <= target_fill:
            lo = mid
        else:
            hi = mid
    scale = lo
    actual_fill = projected_fill(centered, scale, c2ws, fov_deg)
    return center, scale, actual_fill


def apply_global_normalization(imported_objects, center, scale):
    imported_set = set(imported_objects)
    root = bpy.data.objects.new("__InstantMesh_Normalization__", None)
    bpy.context.scene.collection.objects.link(root)

    top_level = [o for o in imported_objects if o.parent is None or o.parent not in imported_set]
    for obj in top_level:
        world = obj.matrix_world.copy()
        obj.parent = root
        obj.matrix_world = world

    update_root_normalization(root, center, scale)
    return root


def update_root_normalization(root, center, scale):
    root.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(Vector((-center[0], -center[1], -center[2])))
    bpy.context.view_layer.update()


def refine_scale_with_alpha(root, center, init_scale, cam, c2ws, probe_dir, target_fill,
                            probe_resolution=128, auto_scale_iters=3,
                            alpha_threshold=0.02, engine="eevee", samples=64,
                            exposure=0.0, keep_probes=False, world_strength=0.15):
    """
    Refine scale using ACTUAL rendered alpha silhouettes rather than projected vertices.
    This better matches the final visible object extent.
    """
    setup_world_for_rgba(world_strength=world_strength)
    setup_render(probe_resolution, engine, samples, exposure=exposure)

    scale = float(init_scale)
    history = []
    probe_dir.mkdir(parents=True, exist_ok=True)

    for it in range(max(1, int(auto_scale_iters))):
        update_root_normalization(root, center, scale)
        fills = []
        for i, c2w in enumerate(c2ws):
            path = probe_dir / f"iter{it:02d}_view{i:02d}.png"
            render_view_rgba(cam, c2w, path)
            rgba = load_image_pixels_rgba(path)
            fills.append(alpha_fill_ratio(rgba, alpha_threshold=alpha_threshold))
        current_fill = float(max(fills)) if fills else 0.0
        history.append({
            "iter": it,
            "scale": scale,
            "fills": fills,
            "max_fill": current_fill,
        })

        if current_fill <= 1e-6:
            break

        # Direct multiplicative correction is a good approximation when centered.
        scale_factor = target_fill / current_fill
        # Clamp to avoid unstable overshoot if the approximation is imperfect.
        scale_factor = float(np.clip(scale_factor, 0.7, 1.6))
        new_scale = scale * scale_factor

        if abs(new_scale - scale) / max(scale, 1e-8) < 0.01:
            scale = new_scale
            break
        scale = new_scale

    update_root_normalization(root, center, scale)

    if not keep_probes:
        shutil.rmtree(probe_dir, ignore_errors=True)

    return scale, history


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_camera_files(asset_dir, c2ws, fov_deg):
    K = get_normalized_K(fov_deg)
    Ks = np.repeat(K[None, :, :], 6, axis=0)
    np.savez(
        asset_dir / "instantmesh_cameras.npz",
        c2ws=c2ws.astype(np.float32),
        Ks=Ks.astype(np.float32),
        azimuths_deg=AZIMUTHS_DEG.astype(np.float32),
        elevations_deg=ELEVATIONS_DEG.astype(np.float32),
    )
    flat = c2ws.reshape(6, 16)[:, :12]
    intr = np.stack([Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]], axis=-1)
    cameras16 = np.concatenate([flat, intr], axis=-1)
    np.save(asset_dir / "instantmesh_cameras_16d.npy", cameras16.astype(np.float32))


def unique_asset_name(path, used):
    base = path.stem
    name = base
    k = 1
    while name in used:
        name = f"{base}_{k:03d}"
        k += 1
    used.add(name)
    return name


# -----------------------------------------------------------------------------
# Main render path
# -----------------------------------------------------------------------------

def render_asset(model_path, output_root, args, asset_name):
    print("\n" + "=" * 80)
    print(f"[InstantMesh render] {model_path}")
    print("=" * 80)

    clear_scene()
    imported = import_model(model_path)
    remove_imported_cameras_and_lights()

    meshes = mesh_objects()
    if not meshes:
        raise RuntimeError(f"No mesh objects found in: {model_path}")
    if not args.no_smooth:
        set_smooth_shading(meshes)
    if args.material_override:
        apply_material_override(meshes, color=args.override_color, roughness=args.override_roughness)

    c2ws = get_instantmesh_c2ws(args.radius)
    if args.training_export:
        target_c2ws, target_azimuths, target_elevations = get_training_target_cameras(
            args.target_views,
            radius=args.radius,
            elevation_min=args.target_elevation_min,
            elevation_max=args.target_elevation_max,
            azimuth_offset=args.target_azimuth_offset,
        )
        scale_probe_c2ws = choose_scale_probe_cameras(
            c2ws, target_c2ws, max_target_views=args.scale_probe_target_views
        )
    else:
        target_c2ws = np.empty((0, 4, 4), dtype=np.float64)
        target_azimuths = np.empty((0,), dtype=np.float64)
        target_elevations = np.empty((0,), dtype=np.float64)
        scale_probe_c2ws = c2ws

    asset_dir = output_root / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)

    # 1) Coarse normalization from projected mesh vertices.
    verts_world = world_mesh_vertices(meshes, max_vertices=args.max_vertices)
    center, coarse_scale, coarse_fill = solve_uniform_scale(verts_world, scale_probe_c2ws, args.fov, args.fill)
    root = apply_global_normalization(imported, center, coarse_scale)

    # 2) Configure illumination and refine scale using actual rendered alpha silhouettes.
    # Diagnostic override is emission-only and therefore needs NO scene lighting.
    effective_world_strength = 0.0 if args.material_override else float(args.world_strength)
    setup_world_for_rgba(world_strength=effective_world_strength)
    if args.material_override:
        print("  [diagnostic] emission material override enabled; scene lights disabled")
    else:
        setup_lighting(
            key_energy=args.key_energy,
            fill_energy=args.fill_energy,
            rim_energy=args.rim_energy,
            sun_energy=args.sun_energy,
        )
    setup_render(args.resolution, args.engine, args.samples, exposure=args.render_exposure)
    cam = make_camera(c2ws[0], args.fov)

    probe_dir = asset_dir / "alpha_probe"
    refined_scale, probe_history = refine_scale_with_alpha(
        root=root,
        center=center,
        init_scale=coarse_scale,
        cam=cam,
        c2ws=scale_probe_c2ws,
        probe_dir=probe_dir,
        target_fill=args.fill,
        probe_resolution=args.probe_resolution,
        auto_scale_iters=args.auto_scale_iters,
        alpha_threshold=args.alpha_threshold,
        engine=args.engine,
        samples=args.samples,
        exposure=args.render_exposure,
        keep_probes=args.keep_probes,
        world_strength=effective_world_strength,
    )
    update_root_normalization(root, center, refined_scale)

    # Ensure final render uses the requested final resolution and exposure.
    # Lighting was already created once above. Do NOT call setup_lighting() again,
    # otherwise Blender adds Key.001 / Fill.001 / Rim.001 / Sun.001 and roughly
    # doubles the illumination for the final renders.
    setup_world_for_rgba(world_strength=effective_world_strength)
    setup_render(args.resolution, args.engine, args.samples, exposure=args.render_exposure)

    rgba_raw_dir = asset_dir / "rgba_raw"
    if args.save_rgba:
        rgba_raw_dir.mkdir(parents=True, exist_ok=True)

    composited_views = []
    final_view_fills = []
    background_rgb_samples = []
    for i, c2w in enumerate(c2ws):
        raw_path = rgba_raw_dir / f"{i:02d}.png" if args.save_rgba else asset_dir / f".__tmp_rgba_{i:02d}.png"
        final_path = asset_dir / f"{i:02d}.png"
        print(f"  view {i}: az={AZIMUTHS_DEG[i]:.0f}°, el={ELEVATIONS_DEG[i]:.0f}° -> {final_path}")
        render_view_rgba(cam, c2w, raw_path)
        raw_rgba = load_image_pixels_rgba(raw_path)
        final_rgba = composite_rgba_to_exact_white(raw_rgba)
        save_image_pixels_rgba(final_path, final_rgba)
        composited_views.append(final_rgba)
        final_view_fills.append(alpha_fill_ratio(raw_rgba, alpha_threshold=args.alpha_threshold))
        background_rgb_samples.append(final_rgba[0, 0, :3].tolist())
        if not args.save_rgba and raw_path.exists():
            raw_path.unlink()

    grid_path = asset_dir / "instantmesh_grid.png"
    save_grid_from_view_arrays(composited_views, grid_path, args.resolution)
    save_camera_files(asset_dir, c2ws, args.fov)

    training_manifest = None
    if args.training_export:
        # Preserve final RGB appearance configuration, then render numerical
        # supervision passes through temporary view-layer material overrides.
        setup_world_for_rgba(world_strength=effective_world_strength)
        setup_render(args.resolution, args.engine, args.samples, exposure=args.render_exposure)
        training_manifest = export_training_dataset(
            asset_dir=asset_dir,
            cam=cam,
            input_c2ws=c2ws,
            target_c2ws=target_c2ws,
            target_azimuths=target_azimuths,
            target_elevations=target_elevations,
            args=args,
        )

    metadata = {
        "source_model": str(model_path.resolve()),
        "asset_name": asset_name,
        "camera": {
            "azimuths_deg": AZIMUTHS_DEG.tolist(),
            "elevations_deg": ELEVATIONS_DEG.tolist(),
            "radius": args.radius,
            "fov_deg": args.fov,
            "convention": "OpenGL/Blender c2w: +X right, +Y up, camera looks -Z, world up +Z",
        },
        "render": {
            "resolution_per_view": args.resolution,
            "grid_width": 2 * args.resolution,
            "grid_height": 3 * args.resolution,
            "grid_order": [[0, 1], [2, 3], [4, 5]],
            "background": "exact_pure_white_after_rgba_composite",
            "engine": args.engine,
            "render_exposure": float(args.render_exposure),
            "world_strength": float(effective_world_strength),
            "lighting": (
                "disabled_emission_override" if args.material_override else {
                    "key_energy": float(args.key_energy),
                    "fill_energy": float(args.fill_energy),
                    "rim_energy": float(args.rim_energy),
                    "sun_energy": float(args.sun_energy),
                }
            ),
        },
        "normalization": {
            "original_center_world": center.tolist(),
            "coarse_scale_from_vertices": float(coarse_scale),
            "coarse_projected_vertex_fill": float(coarse_fill),
            "refined_scale_from_alpha": float(refined_scale),
            "requested_max_silhouette_fill": float(args.fill),
            "final_view_alpha_fills": final_view_fills,
            "final_max_alpha_fill": float(max(final_view_fills) if final_view_fills else 0.0),
            "alpha_threshold": float(args.alpha_threshold),
            "probe_history": probe_history,
            "note": "One shared 3D transform was applied before rendering. No per-view crop/recenter/scale was used.",
        },
        "material_override": {
            "enabled": bool(args.material_override),
            "type": "unlit_emission" if args.material_override else None,
            "color": [float(c) for c in args.override_color],
            "emission_strength": 1.0 if args.material_override else None,
        },
        "debug": {
            "save_rgba": bool(args.save_rgba),
            "raw_rgba_dir": str(rgba_raw_dir) if args.save_rgba else None,
            "keep_probes": bool(args.keep_probes),
            "probe_dir": str(probe_dir) if args.keep_probes else None,
            "background_rgb_samples_top_left": background_rgb_samples,
        },
        "training_export": {
            "enabled": bool(args.training_export),
            "target_views": int(args.target_views) if args.training_export else 0,
            "training_dir": str(asset_dir / "training") if args.training_export else None,
            "depth_scale": float(args.depth_scale) if args.training_export else None,
            "manifest": training_manifest,
        },
        "instantmesh_input_note": "Use instantmesh_grid.png directly. Do not independently crop, recenter, or rescale the six rendered views before reconstruction.",
    }
    with open(asset_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved grid: {grid_path}")
    print(f"  Coarse scale: {coarse_scale:.6f}")
    print(f"  Refined scale: {refined_scale:.6f}")
    print(f"  Final max alpha fill: {max(final_view_fills) if final_view_fills else 0.0:.4f}")
    return grid_path


# -----------------------------------------------------------------------------
# File collection / main
# -----------------------------------------------------------------------------

def collect_models(input_path: Path, recursive: bool):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported input extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    pattern = "**/*" if recursive else "*"
    files = [p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def main():
    args = parse_args()
    print("=" * 80)
    print(f"[SCRIPT] {Path(__file__).resolve()}")
    print(f"[VERSION] {SCRIPT_VERSION}")
    print(f"[BLENDER] {bpy.app.version_string}")
    print("=" * 80)
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not (0.1 <= args.fill < 1.0):
        raise ValueError("--fill must be in [0.1, 1.0).")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive.")
    if args.radius <= 0:
        raise ValueError("--radius must be positive.")
    if args.probe_resolution <= 0:
        raise ValueError("--probe_resolution must be positive.")
    if args.auto_scale_iters <= 0:
        raise ValueError("--auto_scale_iters must be positive.")
    if not (0.0 <= args.alpha_threshold < 1.0):
        raise ValueError("--alpha_threshold must be in [0,1).")
    if args.world_strength < 0.0:
        raise ValueError("--world_strength must be non-negative.")
    for name in ("key_energy", "fill_energy", "rim_energy", "sun_energy"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative.")
    if len(args.override_color) != 3:
        raise ValueError("--override_color must have exactly 3 values.")
    if args.target_views < 0:
        raise ValueError("--target_views must be >= 0.")
    if args.target_elevation_min >= args.target_elevation_max:
        raise ValueError("--target_elevation_min must be smaller than --target_elevation_max.")
    if args.depth_scale <= 0:
        raise ValueError("--depth_scale must be positive.")
    if args.scale_probe_target_views < 0:
        raise ValueError("--scale_probe_target_views must be >= 0.")

    models = collect_models(input_path, args.recursive)
    if not models:
        raise RuntimeError(f"No supported 3D models found under {input_path}")

    print(f"Found {len(models)} model(s).")
    print("InstantMesh camera order:")
    for i in range(6):
        print(f"  {i}: azimuth={AZIMUTHS_DEG[i]:.0f}°, elevation={ELEVATIONS_DEG[i]:.0f}°")

    used_names = set()
    failures = []
    for model_path in models:
        asset_name = unique_asset_name(model_path, used_names)
        try:
            render_asset(model_path, output_root, args, asset_name)
        except Exception as exc:
            failures.append((str(model_path), repr(exc)))
            print(f"[ERROR] {model_path}: {exc}")

    if failures:
        failure_path = output_root / "render_failures.json"
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"\nCompleted with {len(failures)} failure(s). See {failure_path}")
        raise RuntimeError(f"{len(failures)} asset(s) failed to render.")

    print("\nAll assets rendered successfully.")


if __name__ == "__main__":
    main()
