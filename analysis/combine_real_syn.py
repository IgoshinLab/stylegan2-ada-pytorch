import numpy as np
from PIL import Image
import os


def combine_real_syn(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for img_folder in os.listdir(img_dir):
        img1 = Image.open(os.path.join(img_dir, img_folder, "target.png"))
        img2 = Image.open(os.path.join(img_dir, img_folder, "proj.png"))
        img_new = np.concatenate([np.array(img1), np.array(img2)], axis=1)
        Image.fromarray(img_new, 'L').save(f'{out_dir}/{img_folder}.png')


combine_real_syn("/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/syn_imgs2/bp",
                 "/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/syn_imgs2/bp_imgs")