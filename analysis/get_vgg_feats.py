import dnnlib
import torch
import os
import tifffile
import cv2
import numpy as np


def get_vgg16_feats(img_dir, feat_dir):
    os.makedirs(feat_dir, exist_ok=True)
    device = torch.device('cuda')
    url = '/mnt/data/feature_extraction/featmodels/stylegan2-ada-pytorch/models/vgg16.pt'
    vgg16 = torch.jit.load(url).eval().to(device)
    for img_file in os.listdir(img_dir):
        img = tifffile.TiffFile(os.path.join(img_dir, img_file)).asarray()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = torch.tensor(img.transpose([2, 0, 1])).unsqueeze(0).to(device).to(torch.float32)
        img_lpips = vgg16(img, resize_images=False, return_lpips=True)
        img_feat = vgg16(img, resize_images=True, return_features=True)
        folder_name = img_file.split(".")[0]
        feat_folder = os.path.join(feat_dir, folder_name)
        os.makedirs(feat_folder, exist_ok=True)
        np.savez(f'{feat_folder}/projected_rv.npz', w=img_lpips.unsqueeze(0).cpu().numpy(), feat=img_feat.unsqueeze(0).cpu().numpy())


get_vgg16_feats("/mnt/data/feature_extraction/data/Hist_match/clahe_crop",
                "/mnt/data/feature_extraction/data/Hist_match/clahe_feats")