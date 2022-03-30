import numpy as np


def feat_dist(featdir1, featdir2):
    feat1 = np.load(featdir1)['w']
    feat2 = np.load(featdir2)['w']
    difference = np.square(feat1 - feat2).sum()
    return print("The sum of square difference is %f" % difference)


feat_dist("/mnt/data/feature_extraction/data/Sorted_all/images_clahe_crop_feats/Branching||AG1111_081317_534/projected_w.npz",
          "/mnt/data/feature_extraction/data/Sorted_all/images_clahe_crop_feats/Branching||AG1111_081317_534_1/projected_w.npz")