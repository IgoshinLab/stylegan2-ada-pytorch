import os
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE


def clustering(proj_dir):
    legends = {}
    features = []
    all_legends = []
    for i, file_dir in enumerate(os.listdir(proj_dir)):
        feat_name = file_dir.split("||")[0]
        if feat_name in legends:
            legends[feat_name].append(i)
        else:
            legends[feat_name] = [i]
        all_legends.append(feat_name[:5])
        feature = np.load(os.path.join(proj_dir, file_dir, "projected_w.npz"))['w'][0, -1, :]
        features.append(feature)
    features = np.array(features)
    labels = np.zeros(np.shape(features)[0])
    for i, legend in enumerate(legends.keys()):
        labels[legends[legend]] = i
    pca = LDA(n_components=3).fit_transform(features, labels)
    #pca = TSNE(n_components=3).fit_transform(features)
    cm = plt.get_cmap('tab10')
    legend_keys = ["Branching", "Elongated", #"Elongated_or_branching",
                   "Core_WT_Higher_density", "Core_WT_Lower_density", # "Core_Wt",
                   "Small_fb_higher_density", "Small_fb_lower_density",
                   #"Irregular_fb_1", "Irregular_fb_2",
                   "Frz-like", "No_FB_plain", "Phenotype_2"]
    NUM_COLORS = len(legend_keys)
    for j in range(3):
        legend_list = []
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
        for i, legend in enumerate(legend_keys):
            ax.scatter(pca[legends[legend], j % 3], pca[legends[legend], (j + 1) % 3])
            ax.set_xlabel("PC%d" % (j % 3 + 1))
            ax.set_ylabel("PC%d" % ((j + 1) % 3 + 1))
            legend_list.append(legend)
        ax.legend(legend_list, bbox_to_anchor=(1.01, 1), ncol=1)
        fig.tight_layout()
        plt.show()

    linked = linkage(features, 'single')
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=all_legends,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()
    return 0

clustering("/mnt/data/feature_extraction/featmodels/stylegan2-ada-pytorch/out_full")



