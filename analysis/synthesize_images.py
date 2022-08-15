import torch
import dnnlib
import legacy
import numpy as np
import PIL
import os


def synthesize_images(network_pkl, featdir, imgdir, w_avg_samples=100):
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore

    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    synth_images = G.synthesis(w_samples, noise_mode='random')
    w_numpy = w_samples.cpu().numpy().astype(np.float32)  # [N, 1, C]
    synth_images = (synth_images + 1) * (255 / 2)
    synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    os.makedirs(featdir, exist_ok=True)
    os.makedirs(imgdir, exist_ok=True)
    for i in range(w_numpy.shape[0]):
        np.savez(f'{featdir}/{i}.npz', w=w_numpy[i, :])
        PIL.Image.fromarray(synth_images[i, :], 'RGB').save(f'{imgdir}/{i}.tif')



synthesize_images("/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/network-snapshot-001600.pkl",
                  "/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/syn_imgs2/feats",
                  "/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10-selecteddata/syn_imgs2/imgs")