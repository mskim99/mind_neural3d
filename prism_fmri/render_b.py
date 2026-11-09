# B안 렌더 (11-09, 5090 Blender 4.2.3): ShapeNetCore v2 물체 → 자극 영상 프레임 0과 같은 시점 1장 (흰 배경)
#  카메라 = fMRI-Shape camera_pose/000: 원점을 봄, 거리 1.5, 고도 30°, 방위각 0°, z 위 (prep_shape_targets.py와 같은 해석)
#  시야각(fov) · 물체 방향(yaw)은 원문에 없음 → calib 모드로 train 물체 실루엣을 자극 프레임 0과 맞춰 정함
#  물체 정규화: bbox 중심 원점, 가장 긴 변 1 (render_eval.py와 같음)
#  calib : 물체마다 yaw × fov 격자를 Workbench 실루엣(투명 배경)으로 → <out>/<uid>/y{yaw}_f{fov}.png
#  render: Cycles GPU, 흰 배경, <out>/<uid>/0.png (1024²)
# 사용: blender -b -P render_b.py -- --mode calib --root <shapenet_b> --list <uid 목록> --out <폴더> --fovs 20,24,... --yaws 0,90,180,270
#       blender -b -P render_b.py -- --mode render --root <shapenet_b> --list <uid 목록> --out <폴더> --fov F --yaw Y [--shard k/n]
import bpy, sys, os, math, argparse, time
from mathutils import Vector
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['calib', 'render'], required=True)
ap.add_argument('--root', required=True); ap.add_argument('--list', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--fov', type=float, default=30.0); ap.add_argument('--yaw', type=float, default=0.0)
ap.add_argument('--fovs', default=''); ap.add_argument('--yaws', default='0,90,180,270')
ap.add_argument('--dist', type=float, default=1.5); ap.add_argument('--elev', type=float, default=30.0); ap.add_argument('--az', type=float, default=0.0)
ap.add_argument('--res', type=int, default=1024); ap.add_argument('--samples', type=int, default=64)
ap.add_argument('--light', type=float, default=3.0)   # 월드(환경) 조명 세기 — 흰 하늘빛 + 해 1개
ap.add_argument('--sun', type=float, default=2.0)
ap.add_argument('--shard', default='0/1')
a = ap.parse_args(argv)
ids = [l.strip() for l in open(a.list) if l.strip()]
k, n = map(int, a.shard.split('/')); ids = ids[k::n]
sc = bpy.context.scene; R = sc.render

def clear():
    for o in list(bpy.data.objects):
        if o.type == 'MESH' or o.type == 'EMPTY': bpy.data.objects.remove(o, do_unlink=True)
    for b in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for x in list(b):
            if x.users == 0: b.remove(x)

def setup():
    for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
    e, z = math.radians(a.elev), math.radians(a.az)
    cam.location = (a.dist * math.cos(e) * math.cos(z), a.dist * math.cos(e) * math.sin(z), a.dist * math.sin(e))
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    R.resolution_x = R.resolution_y = a.res; R.resolution_percentage = 100
    return cam

cam = setup()
if a.mode == 'calib':
    R.engine = 'BLENDER_WORKBENCH'; R.film_transparent = True; R.image_settings.color_mode = 'RGBA'
else:
    R.engine = 'CYCLES'; sc.cycles.device = 'GPU'; sc.cycles.samples = a.samples; sc.cycles.use_denoising = True
    pr = bpy.context.preferences.addons['cycles'].preferences; pr.compute_device_type = 'OPTIX'; pr.get_devices()
    for d in pr.devices: d.use = d.type != 'CPU'
    R.film_transparent = False; R.image_settings.color_mode = 'RGB'
    sc.view_settings.view_transform = 'Standard'
    w = sc.world or bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (1, 1, 1, 1); bg.inputs[1].default_value = a.light
    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sc.collection.objects.link(sun)
    sun.data.energy = a.sun; sun.rotation_euler = (math.radians(40), 0, math.radians(30))

def load(uid):
    clear()
    f = os.path.join(a.root, uid, 'models', 'model_normalized.obj')
    bpy.ops.wm.obj_import(filepath=f)                    # 기본: ShapeNet y 위 → Blender z 위
    ms = [o for o in sc.objects if o.type == 'MESH']
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for o in ms:
        for c in o.bound_box:
            p = o.matrix_world @ Vector(c); lo = Vector(map(min, lo, p)); hi = Vector(map(max, hi, p))
    s = 1.0 / max(hi - lo); ctr = (lo + hi) / 2
    root = bpy.data.objects.new('root', None); sc.collection.objects.link(root)
    for o in ms:
        o.parent = root; o.matrix_parent_inverse = root.matrix_world.inverted()
    root.location = -ctr * s; root.scale = (s, s, s)
    bpy.context.view_layer.update()
    root2 = bpy.data.objects.new('yaw', None); sc.collection.objects.link(root2); root.parent = root2
    return root2

t0 = time.time()
for i, uid in enumerate(ids):
    od = os.path.join(a.out, uid.replace('/', '_'))
    if a.mode == 'render' and os.path.exists(os.path.join(od, '0.png')): continue
    os.makedirs(od, exist_ok=True)
    try: yawobj = load(uid)
    except Exception as ex: print('LOAD_FAIL', uid, ex, flush=True); continue
    if a.mode == 'calib':
        for y in map(float, a.yaws.split(',')):
            yawobj.rotation_euler = (0, 0, math.radians(y))
            for fv in map(float, a.fovs.split(',')):
                cam.data.angle = math.radians(fv); R.filepath = os.path.join(od, f'y{int(y)}_f{fv:g}.png')
                bpy.ops.render.render(write_still=True)
    else:
        yawobj.rotation_euler = (0, 0, math.radians(a.yaw)); cam.data.angle = math.radians(a.fov)
        R.filepath = os.path.join(od, '0.png'); bpy.ops.render.render(write_still=True)
    if (i + 1) % 20 == 0 or i == 0: print(f'[{a.mode} {i + 1}/{len(ids)}] {uid} {time.time() - t0:.0f}s', flush=True)
print('RENDER_B_DONE', flush=True)
