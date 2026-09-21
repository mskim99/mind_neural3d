from torch.utils.data import Dataset
import os
from pathlib import Path

import numpy as np
import torch
import mne
import sys
import open3d as o3d
import pandas as pd
from PIL import Image


class AllDataFeatureTwoEEG(Dataset):
    """
    Neuro-3D EEG dataset loader for EEG -> canonical Zero123++/InstantMesh 6-view training.

    This keeps the original egg_dataset.py behavior for:
      - train/test object split
      - EEG loading
      - CLIP text/video features
      - subject / class / object / trial indexing

    The only intentional target-side change is:
      - old: sample `num_frames` frames from video_new/<name>.mp4
      - new: load exactly six renderer-generated views 00.png ... 05.png

    Expected rendered-view layout (either candidate is accepted):
        <rendered_view_path>/<name>/00.png ... 05.png
    or
        <rendered_view_path>/<name[3:]>/00.png ... 05.png

    The order is NEVER shuffled because Zero123++ / InstantMesh assumes:
        0 | 1
        2 | 3
        4 | 5
    """

    def __init__(
        self,
        data_path,
        sub_list,
        train=True,
        time_len=250,
        test_mean=True,
        aug_data=False,
        point_path='',
        num_frames=6,
        rendered_view_path=None,
        rendered_image_size=320,
        strict_rendered_views=True,
    ):
        self.data_path = data_path
        self.sub_list = sub_list
        self.train = train
        self.time_len = time_len
        self.test_mean = test_mean
        self.aug_data = aug_data

        if int(num_frames) != 6:
            raise ValueError(
                f"egg_dataset_ext requires exactly 6 canonical views, got num_frames={num_frames}."
            )
        self.num_frames = 6
        self.rendered_image_size = int(rendered_image_size)
        self.strict_rendered_views = bool(strict_rendered_views)

        if rendered_view_path is None:
            rendered_view_path = os.path.join(self.data_path, 'eeg3d_instantmesh_views')
        self.rendered_view_path = Path(rendered_view_path).expanduser()

        all_name_list = sorted(os.listdir(self.data_path + '/video_new/'))
        self.name_list = []
        if train:
            remove_name = ['08', '09']
        else:
            remove_name = ['00', '01', '02', '03', '04', '05', '06', '07']

        for name in all_name_list:
            if name[-4:] != '.mp4':
                continue
            if name[-6:-4] in remove_name:
                continue
            self.name_list.append(name[:-4])

        excel_file = self.data_path + 'color_label.xlsx'
        read_df = pd.read_excel(excel_file)
        name2color_label = {}
        for index, row in read_df.iterrows():
            name2color_label[row['name']] = int(row['label'])

        self.name_list = np.array(self.name_list).reshape(72, -1)
        self.point_data = np.zeros(
            (self.name_list.shape[0], self.name_list.shape[1], 8192, 6)
        )
        self.color_label = np.zeros(
            (self.name_list.shape[0], self.name_list.shape[1])
        )

        if point_path == '':
            point_path = self.data_path + 'point_cloud_simple/'
            for ii in range(self.name_list.shape[0]):
                for jj in range(self.name_list.shape[1]):
                    key = self.name_list[ii][jj][3:]
                    self.point_data[ii, jj] = np.load(point_path + key + '.npy')
                    self.color_label[ii, jj] = name2color_label[key]

            for ii in range(self.point_data.shape[0]):
                for jj in range(self.point_data.shape[1]):
                    self.point_data[ii, jj] = self.pc_norm(self.point_data[ii, jj])
        else:
            app_ss = point_path.split('-')[-1][:-1]
            name_app_dir = {}
            for name_one in os.listdir(point_path):
                name_two = name_one.split('-')[1]
                name_app_dir[name_two[3:]] = name_two[:3]

            for ii in range(self.name_list.shape[0]):
                for jj in range(self.name_list.shape[1]):
                    key = self.name_list[ii][jj][3:]
                    _ = np.load(self.data_path + 'point_cloud_simple/' + key + '.npy')
                    _ = f'{point_path}{app_ss}-{name_app_dir[key]}{key}-best1.ply'
                    self.color_label[ii, jj] = name2color_label[key]

        self.eeg_data, self.eeg_data2 = self.load_eeg()
        self.cls_num = 72

        if not self.train:
            if self.test_mean:
                self.eeg_data = np.mean(self.eeg_data, axis=3, keepdims=True)
                self.eeg_data2 = np.mean(self.eeg_data2, axis=3, keepdims=True)
                self.obj_num, self.trails_num = 2, 1
            else:
                self.obj_num, self.trails_num = 2, 4
        else:
            self.obj_num, self.trails_num = 8, 2

        clip_feature_name = self.data_path + 'clip_feature.pth'
        clip_feature_gray_name = self.data_path + 'clip_feature_gray.pth'

        self.clip_features = torch.load(clip_feature_name)
        self.clip_features_gray = torch.load(clip_feature_gray_name)

        for key in self.clip_features.keys():
            one_fea = self.clip_features[key]['point']
            one_fea = one_fea / 8.0
            self.clip_features[key]['point'] = one_fea

        for key in self.clip_features_gray.keys():
            one_fea = self.clip_features_gray[key]['point']
            one_fea = one_fea / 8.0
            self.clip_features_gray[key]['point'] = one_fea

        print(
            f'name:{self.name_list.shape}, point:{self.point_data.shape}, '
            f'eegdata:{self.eeg_data.shape}, eegdata2:{self.eeg_data2.shape}'
        )
        print(f'[egg_dataset_ext] rendered_view_path={self.rendered_view_path}')
        print(
            f'[egg_dataset_ext] canonical views=6, '
            f'image_size={self.rendered_image_size}'
        )

        for ii in range(self.name_list.shape[0]):
            for jj in range(self.name_list.shape[1]):
                if self.name_list[ii, jj][3:] not in self.clip_features.keys():
                    print(self.name_list[3:])

        self.txt_features = torch.zeros(
            (self.name_list.shape[0], self.name_list.shape[1], 1024)
        )
        self.color_video_features = torch.zeros(
            (self.name_list.shape[0], self.name_list.shape[1], 1024)
        )
        self.color_point_features = torch.zeros(
            (self.name_list.shape[0], self.name_list.shape[1], 768)
        )
        self.gray_video_features = torch.zeros(
            (self.name_list.shape[0], self.name_list.shape[1], 1024)
        )
        self.gray_point_features = torch.zeros(
            (self.name_list.shape[0], self.name_list.shape[1], 768)
        )

        for ii in range(self.name_list.shape[0]):
            for jj in range(self.name_list.shape[1]):
                name = self.name_list[ii, jj]
                key = name[3:]
                self.txt_features[ii, jj] = self.clip_features[key]['text']
                self.color_video_features[ii, jj] = self.clip_features[key]['video']
                self.color_point_features[ii, jj] = self.clip_features[key]['point']
                self.gray_video_features[ii, jj] = self.clip_features_gray[key]['video']
                self.gray_point_features[ii, jj] = self.clip_features_gray[key]['point']

        if self.strict_rendered_views:
            self._validate_rendered_dataset()

        self._validate_name_table_structure()

    def _validate_name_table_structure(self):
        """
        Validate that every EEG class row maps to one stable 3-character
        dataset prefix. This catches accidental reshape/sort drift before
        training/inference.
        """
        prefixes = []
        for cls_index in range(self.name_list.shape[0]):
            row = [str(x) for x in self.name_list[cls_index]]
            row_prefixes = sorted(set(x[:3] for x in row))
            if len(row_prefixes) != 1:
                raise RuntimeError(
                    f"Class row {cls_index} mixes prefixes: {row_prefixes}. "
                    f"name_list sorting/reshape is not a valid EEG-label mapping."
                )
            prefixes.append(row_prefixes[0])

        if len(set(prefixes)) != len(prefixes):
            raise RuntimeError(
                "Duplicate 3-character class prefixes found across EEG class rows. "
                "Cannot use the current name_list reshape as a unique class mapping."
            )

        print(
            f"[egg_dataset_ext] verified {len(prefixes)} EEG class rows "
            f"with unique stable filename prefixes."
        )

    def pc_norm(self, pc):
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        other_feature = (other_feature - 0.5) * 2
        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def load_eeg(self):
        eeg_path = self.data_path + 'EEGdata/'
        eeg_data_all, eeg_data_all2 = [], []

        for sub in self.sub_list:
            if self.train:
                sub_eeg_data_name2 = f"{eeg_path}{sub}/{sub}_train_data_1s_250Hz.npy"
                sub_eeg_data_name = f"{eeg_path}{sub}/{sub}_train_data_6s_100Hz.npy"
            else:
                sub_eeg_data_name2 = f"{eeg_path}{sub}/{sub}_test_data_1s_250Hz.npy"
                sub_eeg_data_name = f"{eeg_path}{sub}/{sub}_test_data_6s_100Hz.npy"

            sub_eeg_data = np.load(sub_eeg_data_name, mmap_mode='r')
            sub_eeg_data2 = np.load(sub_eeg_data_name2, mmap_mode='r')
            eeg_data_all.append(sub_eeg_data[np.newaxis, :, :, :, :, :])
            eeg_data_all2.append(sub_eeg_data2[np.newaxis, :, :, :, :, :])

        eeg_data_all = np.concatenate(eeg_data_all, axis=0)
        eeg_data_all2 = np.concatenate(eeg_data_all2, axis=0)
        return eeg_data_all, eeg_data_all2

    def _render_dir_candidates(self, name):
        candidates = [
            self.rendered_view_path / str(name),
            self.rendered_view_path / str(name)[3:],
        ]
        unique = []
        seen = set()
        for p in candidates:
            s = str(p)
            if s not in seen:
                unique.append(p)
                seen.add(s)
        return unique

    def _resolve_render_dir(self, name):
        required = [f"{i:02d}.png" for i in range(6)]

        for candidate in self._render_dir_candidates(name):
            if candidate.is_dir() and all((candidate / f).is_file() for f in required):
                return candidate

        tried = "\n  ".join(str(x) for x in self._render_dir_candidates(name))
        raise FileNotFoundError(
            f"Could not find six rendered views for Neuro-3D sample '{name}'.\n"
            f"Tried:\n  {tried}\n"
            f"Expected files: 00.png ... 05.png"
        )

    def _load_rgb_view(self, path):
        with Image.open(path) as im:
            im = im.convert('RGB')
            if im.size != (self.rendered_image_size, self.rendered_image_size):
                im = im.resize(
                    (self.rendered_image_size, self.rendered_image_size),
                    resample=Image.Resampling.BICUBIC,
                )
            arr = np.asarray(im, dtype=np.float32) / 255.0

        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()

    def load_rotation_images(self, name):
        render_dir = self._resolve_render_dir(name)
        views = [
            self._load_rgb_view(render_dir / f"{i:02d}.png")
            for i in range(6)
        ]
        return torch.stack(views, dim=0)

    def _validate_rendered_dataset(self):
        missing = []
        checked = set()

        for name in self.name_list.reshape(-1):
            name = str(name)
            if name in checked:
                continue
            checked.add(name)

            try:
                self._resolve_render_dir(name)
            except FileNotFoundError:
                missing.append(name)

        if missing:
            preview = ", ".join(missing[:20])
            suffix = "" if len(missing) <= 20 else f", ... (+{len(missing)-20})"
            raise FileNotFoundError(
                f"[egg_dataset_ext] {len(missing)} rendered assets are missing "
                f"under {self.rendered_view_path}.\n"
                f"Examples: {preview}{suffix}\n"
                f"Set rendered_view_path to the directory containing the renderer outputs."
            )

        print(
            f"[egg_dataset_ext] validated {len(checked)} rendered assets "
            f"for {'train' if self.train else 'test'} split."
        )

    def __len__(self):
        num = 1
        for ii in range(len(self.eeg_data.shape) - 2):
            num = num * self.eeg_data.shape[ii]
        return num

    def add_noise(self, eeg_data):
        stds = eeg_data.std(dim=1, keepdim=True)
        stds[torch.isnan(stds)] = 0
        noise = torch.randn_like(eeg_data) * stds * 0.2
        return eeg_data + noise

    def __getitem__(self, idx):
        sub_index, sub_other = (
            idx // (self.cls_num * self.obj_num * self.trails_num),
            idx % (self.cls_num * self.obj_num * self.trails_num),
        )
        cls_index, cls_other = (
            sub_other // (self.obj_num * self.trails_num),
            sub_other % (self.obj_num * self.trails_num),
        )
        obj_index, obj_other = (
            cls_other // self.trails_num,
            cls_other % self.trails_num,
        )

        name = self.name_list[cls_index, obj_index]

        if self.aug_data and np.random.rand() > 0.75:
            eeg_data = np.mean(
                self.eeg_data[sub_index, cls_index, obj_index, :],
                axis=0,
            )
        else:
            eeg_data = self.eeg_data[sub_index, cls_index, obj_index, obj_other]

        if self.aug_data and np.random.random() > 0.4:
            eeg_data_new = self.add_noise(torch.from_numpy(eeg_data))
        else:
            eeg_data_new = torch.from_numpy(eeg_data)

        key = name[3:]
        txt_fea = self.clip_features[key]['text']
        color_video_fea = self.clip_features[key]['video']

        rotation_images = self.load_rotation_images(name)

        return {
            'name': str(name),
            # Source of truth for all later file naming/evaluation.
            'label': str(key),
            'class_prefix': str(name)[:3],
            'cls_index': int(cls_index),
            'obj_index': int(obj_index),
            'trial_index': int(obj_other),
            'subject_index': int(sub_index),
            'eeg_data': eeg_data_new,
            'txt_fea': txt_fea,
            'color_video_fea': color_video_fea,
            'rotation_images': rotation_images,
        }
