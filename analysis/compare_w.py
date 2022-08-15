import numpy as np
import os
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error

def compare_w(root_dir1, root_dir2):
    coss = []
    mses = []
    for file in os.listdir(root_dir1):
        feature1 = np.load(os.path.join(root_dir1, file, "projected_w.npz"))
        feature2 = np.load(os.path.join(root_dir2, file, "projected_w.npz"))
        w1 = feature1['w'][0, 0, :]
        w2 = feature2['w'][0, 0, :]
        cos = distance.cosine(w1, w2)
        mse = mean_squared_error(w1, w2)
        print("cos=%f,\t mse=%f" % (cos, mse))
        coss.append(cos)
        mses.append(mse)
    coss = np.array(coss)
    print("cos mean:%f\tstd:%f" % (coss.mean(), coss.std()))
    mses = np.array(mses)
    print("mse mean:%f\tstd:%f" % (mses.mean(), mses.std()))


compare_w("/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/backproject",
          "/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/backproject2")