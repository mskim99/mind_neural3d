# V-8 진단 렌더 (11-28, 총괄 04:47: 반응 판정을 Blender에 묶지 않음) — GLB 텍스처 색을 nvdiffrast로 조명 없이 그림 (meshgen venv, GPU 0, 학습 없음)
#  조건끼리 같은 렌더러로만 비교하기 위한 진단용. 8지표(Blender, 조명 있음)와 다른 렌더러다
#  카메라: 자극 · 평가와 같은 거리 1.5 · FOV 56 · 고도 30°, 방위 0 · 60 · … · 300 (6뷰), 위쪽 +Y (glTF)
#  메시: GLB(trimesh, 한 메시로 합침) → bbox 중심을 원점으로, 가장 긴 변 = 1로 맞춤 (조건 사이 형상이 같으므로 같은 정규화)
#  색: baseColorTexture를 uv로 그대로 (조명 · 그림자 없음), 512², RGBA(배경 알파 0)
# 출력: <out>/<uid>/{0..5}.png
# 사용: python v8_diag_render.py --glb <폴더(<uid>/<uid>.glb)> --out <폴더>
import os, glob, argparse, math, numpy as np, torch, trimesh
import nvdiffrast.torch as dr
from PIL import Image
p = argparse.ArgumentParser(); p.add_argument('--glb', required=True); p.add_argument('--out', required=True); p.add_argument('--res', type=int, default=512)
a = p.parse_args(); dev = 'cuda'; ctx = dr.RasterizeCudaContext(); S = a.res


def persp(fov, near=0.05, far=10):
    f = 1 / math.tan(math.radians(fov) / 2)
    return torch.tensor([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]], dtype=torch.float32)


def lookat(eye):
    eye = np.array(eye, np.float64); z = eye / np.linalg.norm(eye); x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3] = x, y, z; M[:3, 3] = -M[:3, :3] @ eye; return torch.tensor(M, dtype=torch.float32)


P = persp(56)
VIEWS = [lookat([1.5 * math.cos(math.radians(30)) * math.sin(math.radians(az)), 1.5 * math.sin(math.radians(30)), 1.5 * math.cos(math.radians(30)) * math.cos(math.radians(az))]) for az in range(0, 360, 60)]
files = sorted(glob.glob(f'{a.glb}/*/*.glb')); n_ok = 0
for f in files:
    uid = os.path.basename(f)[:-4]; od = f'{a.out}/{uid}'
    if os.path.exists(f'{od}/5.png'): n_ok += 1; continue
    m = trimesh.load(f, force='mesh', process=False)
    tex = m.visual.material.baseColorTexture if hasattr(m.visual.material, 'baseColorTexture') else None
    if tex is None or m.visual.uv is None: print('텍스처 없음', uid, flush=True); continue
    v = np.asarray(m.vertices, np.float32); lo, hi = v.min(0), v.max(0); v = (v - (lo + hi) / 2) / (hi - lo).max()
    pos = torch.from_numpy(np.c_[v, np.ones(len(v), np.float32)]).to(dev); tri = torch.from_numpy(np.asarray(m.faces, np.int32)).to(dev)
    uv = np.asarray(m.visual.uv, np.float32).copy(); uv[:, 1] = 1 - uv[:, 1]; uv = torch.from_numpy(uv).to(dev)
    T = torch.from_numpy(np.asarray(tex.convert('RGB'), np.float32) / 255).to(dev)[None]
    os.makedirs(od, exist_ok=True)
    for k, Vw in enumerate(VIEWS):
        clip = (pos @ (P @ Vw).t().to(dev))[None]
        rast, _ = dr.rasterize(ctx, clip, tri, (S, S))
        tc, _ = dr.interpolate(uv[None], rast, tri)
        col = dr.texture(T.contiguous(), tc, filter_mode='linear')
        al = (rast[..., 3:4] > 0).float(); img = torch.cat([col, al], -1)
        img = dr.antialias(img, rast, clip, tri)[0].flip(0)
        Image.fromarray((img.clamp(0, 1) * 255).round().byte().cpu().numpy(), 'RGBA').save(f'{od}/{k}.png')
    n_ok += 1
print('DIAG_RENDER_DONE', n_ok, '/', len(files), a.out, flush=True)
