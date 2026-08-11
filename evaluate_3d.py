import os
import glob
import torch
import numpy as np
import trimesh
from scipy.linalg import sqrtm
from geomloss import SamplesLoss
import torch.nn as nn
from tqdm import tqdm


# ==========================================================
# 1. Metric Calculation Functions
# ==========================================================
def calc_cd(p1, p2):
    """ Chamfer Distance 연산 """
    p1 = p1.unsqueeze(2)  # [B, N, 1, 3]
    p2 = p2.unsqueeze(1)  # [B, 1, M, 3]

    dist = torch.sum((p1 - p2) ** 2, dim=-1)  # [B, N, M]

    min1_dist, _ = torch.min(dist, dim=2)  # [B, N]
    min2_dist, _ = torch.min(dist, dim=1)  # [B, M]

    cd = torch.mean(min1_dist, dim=1) + torch.mean(min2_dist, dim=1)
    return torch.mean(cd).item()


def calc_emd(p1, p2):
    """ Earth Mover's Distance 연산 (Sinkhorn) """
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    emd_val = loss(p1, p2)
    return torch.mean(emd_val).item()


def calc_fpd(features_real, features_fake):
    """ Fréchet Point Cloud Distance 연산 """
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    mu_fake = np.mean(features_fake, axis=0)
    sigma_fake = np.cov(features_fake, rowvar=False)

    diff = mu_real - mu_fake
    diff_squared = diff.dot(diff)

    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fpd = diff_squared + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fpd)


# ==========================================================
# 2. PointNet Feature Extractor (FPD 측정용)
# ==========================================================
class PointNetExtractor(nn.Module):
    """ FPD 측정을 위해 1024차원 글로벌 특징을 추출하는 단순화된 PointNet """

    def __init__(self):
        super(PointNetExtractor, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [B, 3, N]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # [B, 1024]
        return x


# ==========================================================
# 3. Main Evaluation Pipeline
# ==========================================================
def load_and_sample_mesh(file_path, num_points=2048):
    """ 메쉬 파일을 불러와 표면에서 포인트를 샘플링 """
    mesh = trimesh.load(file_path, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    # 원점 중앙화 및 스케일 정규화 (선택 사항이나 평가 안정성을 위해 권장)
    points = points - np.mean(points, axis=0)
    max_norm = np.max(np.linalg.norm(points, axis=1))
    points = points / max_norm

    return points.astype(np.float32)


def evaluate(pred_dir, gt_dir, num_points=2048, device='cuda'):
    print(f"[*] Evaluation Started")
    print(f" - Predict Dir: {pred_dir}")
    print(f" - GT Dir: {gt_dir}")

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.obj")))

    if len(pred_files) == 0:
        raise ValueError("No .obj files found in the prediction directory.")

    total_cd = 0.0
    total_emd = 0.0

    all_pred_points = []
    all_gt_points = []

    # 3.1 파일 순회 및 CD, EMD 계산
    for pred_path in tqdm(pred_files, desc="Calculating CD & EMD"):
        file_name = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, file_name)

        if not os.path.exists(gt_path):
            print(f"[Warning] GT mesh not found for {file_name}. Skipping...")
            continue

        # 메쉬 로드 및 샘플링
        pred_pts = load_and_sample_mesh(pred_path, num_points)
        gt_pts = load_and_sample_mesh(gt_path, num_points)

        all_pred_points.append(pred_pts)
        all_gt_points.append(gt_pts)

        # 텐서 변환 (Batch 차원 추가)
        p1 = torch.tensor(pred_pts).unsqueeze(0).to(device)  # [1, 2048, 3]
        p2 = torch.tensor(gt_pts).unsqueeze(0).to(device)  # [1, 2048, 3]

        # 배치(1) 단위 지표 누적
        total_cd += calc_cd(p1, p2)
        total_emd += calc_emd(p1, p2)

    valid_count = len(all_pred_points)
    avg_cd = total_cd / valid_count
    avg_emd = total_emd / valid_count

    # 3.2 FPD 계산을 위한 특징 추출
    print("[*] Extracting features for FPD calculation...")

    # 사전 학습된 PointNet 로드
    # (실제 논문 평가 시에는 ShapeNet 등으로 사전 학습된 가중치 경로를 지정해야 정확한 FPD가 도출됩니다)
    pointnet = PointNetExtractor().to(device)
    pointnet.eval()

    # 메모리 효율을 위해 전체를 하나의 텐서로 묶어 처리
    pred_tensor = torch.tensor(np.array(all_pred_points)).transpose(1, 2).to(device)  # [B, 3, N]
    gt_tensor = torch.tensor(np.array(all_gt_points)).transpose(1, 2).to(device)

    with torch.no_grad():
        # 배치 크기가 너무 크면 분할 처리(Chunking)가 필요할 수 있습니다.
        pred_features = pointnet(pred_tensor).cpu().numpy()
        gt_features = pointnet(gt_tensor).cpu().numpy()

    fpd_score = calc_fpd(gt_features, pred_features)

    # 3.3 최종 결과 출력
    print("\n" + "=" * 40)
    print(" 📊 Evaluation Results ")
    print("=" * 40)
    print(f" Tested pairs : {valid_count}")
    print(f" CD (x10^3)   : {avg_cd * 1000:.4f} ↓")  # 통상적으로 보기 좋게 1000을 곱하여 표기합니다
    print(f" EMD          : {avg_emd:.4f} ↓")
    print(f" FPD          : {fpd_score:.4f} ↓")
    print("=" * 40)


if __name__ == '__main__':
    # 테스트할 폴더 경로를 사용자 환경에 맞게 지정합니다
    PRED_DIRECTORY = "./inference_results/meshes"
    GT_DIRECTORY = "/data/jionkim/neuro_3D/gt_meshes"

    evaluate(pred_dir=PRED_DIRECTORY, gt_dir=GT_DIRECTORY)