import os
import argparse
from PIL import Image
import numpy as np
import os.path as osp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate empty Cityscapes-style labelTrainIds labels.')
    parser.add_argument('dataset_path', help='Path to WildPASS2K dataset root')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'], help='Split name')
    return parser.parse_args()

def main():
    args = parse_args()

    img_root = osp.join(args.dataset_path, 'leftImg8bit')
    label_root = osp.join(args.dataset_path, 'gtFine')
    os.makedirs(label_root, exist_ok=True)

    id_txt_path = osp.join(args.dataset_path, f'{args.split}.txt')
    id_txt_lines = []

    count = 0
    cities = sorted(os.listdir(img_root))
    for city in tqdm(cities, desc="Processing cities"):
        city_img_dir = osp.join(img_root, city)
        if not osp.isdir(city_img_dir):
            continue

        city_label_dir = osp.join(label_root, city)
        os.makedirs(city_label_dir, exist_ok=True)

        fnames = sorted(os.listdir(city_img_dir))
        for fname in tqdm(fnames, desc=f"  {city}", leave=False):
            if not fname.endswith('.jpg'):
                continue
            basename = osp.splitext(fname)[0]
            img_id = f'{city}_{basename.zfill(6)}'
            img_filename = f'{img_id}_leftImg8bit.png'
            label_filename = f'{img_id}_gtFine_labelTrainIds.png'

            # Convert image to PNG and save (optional)
            src_img_path = osp.join(city_img_dir, fname)
            dst_img_path = osp.join(city_img_dir, img_filename)
            # if not osp.exists(dst_img_path):
            img = Image.open(src_img_path).convert('RGB')
            img.save(dst_img_path)

            # Create empty label (255)
            label_path = osp.join(city_label_dir, label_filename)
            if not osp.exists(label_path):
                empty_label = np.ones((img.height, img.width), dtype=np.uint8) * 255
                Image.fromarray(empty_label).save(label_path)

            id_txt_lines.append(f'{city}/{img_id}\n')
            count += 1

    with open(id_txt_path, 'w') as f:
        f.writelines(id_txt_lines)

    print(f'\nProcessed {count} images.')
    print(f'{args.split}.txt saved to: {id_txt_path}')


if __name__ == '__main__':
    main()
