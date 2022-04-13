import tifffile
import numpy as np
import os
import cv2


def merge_img_mask(img_dir, mask_dir, out_dir):
    for img_folder in os.listdir(img_dir):
        for img_file in os.listdir(os.path.join(img_dir, img_folder)):
            if os.path.exists(os.path.join(mask_dir, img_folder, img_file)):
                img = tifffile.TiffFile(os.path.join(img_dir, img_folder, img_file)).asarray()
                mask = tifffile.TiffFile(os.path.join(mask_dir, img_folder, img_file)).asarray()
                img = np.array([img, mask, 255 - mask]).transpose([1, 2, 0])
                if not os.path.exists(os.path.join(out_dir, img_folder)):
                    os.makedirs(os.path.join(out_dir, img_folder))
                tifffile.imwrite(os.path.join(out_dir, img_folder, img_file), img)

# without folders
def merge_img_mask2(img_dir, mask_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    normalize = cv2.createCLAHE(clipLimit=40, tileGridSize=(16, 16))
    for img_file in os.listdir(img_dir):
        if mask_dir and os.path.exists(os.path.join(mask_dir, img_file)):
            img = tifffile.TiffFile(os.path.join(img_dir, img_file)).asarray()
            if img.dtype != np.uint8:
                img = np.uint8(img / img.max() * 255)
            img = normalize.apply(img)
            mask = tifffile.TiffFile(os.path.join(mask_dir, img_file)).asarray()
            img = np.array([img, mask, 255 - mask]).transpose([1, 2, 0])
            tifffile.imwrite(os.path.join(out_dir, img_file), img)
        else:
            img = tifffile.TiffFile(os.path.join(img_dir, img_file)).asarray()
            if img.dtype != np.uint8:
                img = np.uint8(img / img.max() * 255)
            img = normalize.apply(img)
            tifffile.imwrite(os.path.join(out_dir, img_file), img)


merge_img_mask2("/mnt/data/feature_extraction/data/Hist_match/original_all",
                None,
                "/mnt/data/feature_extraction/data/Hist_match/clahe")