#!/usr/bin/env python3
"""Preflight MinD-3D++ fMRI-Objaverse E3 GT assets (stdlib only).

Expected train-list rows include an optional header such as ``category,uid`` or
``category,object_path``; object rows are ``category,000-000/<uid>.glb``.
The checker does NOT create view images. It verifies files on disk.
"""

import argparse
import csv
from pathlib import Path


def _header_row(category: str, path_field: str) -> bool:
    """Recognize two-column CSV headers without mistaking them for Objaverse UIDs."""
    category = category.strip().casefold().replace(' ', '_')
    path_field = path_field.strip().casefold().replace(' ', '_')
    category_headers = {'category', 'category_name', 'class', 'class_name'}
    path_headers = {'uid', 'object_uid', 'object_id', 'object_path',
                    'path', 'mesh_path', 'glb_path', 'object'}
    return category in category_headers and path_field in path_headers


def parse_list(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Missing train_list: {path}')

    objects = {}
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        for row_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            parts = next(csv.reader([line])) if ',' in line else line.split()
            parts = [part.strip() for part in parts if part.strip()]
            if len(parts) < 2:
                raise ValueError(f'{path}:{row_number}: expected category,object_path: {line!r}')

            category = ' '.join(parts[:-1])
            rel = parts[-1].replace('\\', '/').lstrip('/')
            if _header_row(category, rel):
                continue
            if not rel.lower().endswith('.glb'):
                raise ValueError(f'{path}:{row_number}: expected a GLB path, got {rel!r}')

            uid = Path(rel).stem
            if not uid:
                raise ValueError(f'{path}:{row_number}: empty UID')
            if uid in objects and objects[uid] != rel:
                raise ValueError(f'Duplicate UID with different paths: {uid}')
            objects[uid] = rel

    if not objects:
        raise ValueError(f'No object records found in {path}')
    return objects


def main():
    parser = argparse.ArgumentParser(description='Check E3 GT training meshes and 6-view images')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--gt_views_root', required=True)
    parser.add_argument('--gt_mesh_root', default='')
    parser.add_argument('--num_views', type=int, default=6)
    args = parser.parse_args()

    if args.num_views < 1:
        parser.error('--num_views must be >= 1')

    objects = parse_list(Path(args.data_path) / 'train_list.txt')
    views_root = Path(args.gt_views_root).expanduser()
    missing_views = []
    count_views = 0
    count_complete = 0
    for uid in objects:
        present = 0
        for index in range(args.num_views):
            view = views_root / uid / f'{index:02d}.png'
            if view.is_file():
                count_views += 1
                present += 1
            else:
                missing_views.append((uid, view))
        if present == args.num_views:
            count_complete += 1

    missing_meshes = []
    if args.gt_mesh_root:
        meshes_root = Path(args.gt_mesh_root).expanduser()
        missing_meshes = [meshes_root / rel for rel in objects.values()
                          if not (meshes_root / rel).is_file()]

    print(f'GT train-list UID count: {len(objects)}')
    print(f'GT six-view images: {count_views}/{len(objects) * args.num_views}')
    print(f'GT objects with complete views: {count_complete}/{len(objects)}')
    print(f'GT views root: {views_root} (exists={views_root.is_dir()})')
    if args.gt_mesh_root:
        print(f'GT GLB meshes: {len(objects)-len(missing_meshes)}/{len(objects)}')

    if missing_views:
        print('Missing view examples:')
        for uid, view in missing_views[:8]:
            print(f'  {uid}: {view}')
    if missing_meshes:
        print('Missing GLB examples:')
        for mesh in missing_meshes[:8]:
            print(f'  {mesh}')

    if missing_views or missing_meshes:
        if count_views == 0:
            print('[HINT] No GT training views were found at this root. '
                  'Render train_list.txt with Blender or correct --gt_views_root.')
        raise SystemExit('GT preflight FAILED: required E3 assets are missing.')
    print('GT preflight PASSED: all training/validation UID assets are present.')


if __name__ == '__main__':
    main()
