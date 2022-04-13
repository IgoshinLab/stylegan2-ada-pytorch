import tifffile
import cv2
import numpy as np
import os


def crop_resize(img_dir, out_dir, x, y, resize=1/4):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for img_file in os.listdir(img_dir):
        img = tifffile.TiffFile(os.path.join(img_dir, img_file)).asarray()
        h, w = np.shape(img)[:2]
        w = int(w * resize)
        h = int(h * resize)
        img = cv2.resize(img, (w, h), cv2.INTER_LANCZOS4)
        center = np.shape(img)[:2]
        x_start = (center[0] - x) // 2
        y_start = (center[1] - y) // 2
        img = img[x_start:x_start+x, y_start:y_start+y]
        tifffile.imwrite(os.path.join(out_dir, img_file), img)


crop_resize("/mnt/data/feature_extraction/data/Hist_match/inputs_collect_all",
            "/mnt/data/feature_extraction/data/Hist_match/hm_crop", 224, 224, resize=1/4)