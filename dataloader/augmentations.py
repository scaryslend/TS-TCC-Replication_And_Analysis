import numpy as np
import torch

def DataTransform(sample, config):
    weak_aug = jitter(scaling(sample, config.augmentation.jitter_scale_ratio), config.augmentation.jitter_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    return weak_aug, strong_aug

def jitter(x, sigma=0.8):

    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=1.1):

    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return x * factor[:, np.newaxis, :]

def permutation(x, max_segments=5, seg_mode='random'):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == 'random':
                split_points = np.sort(np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False) + 1)
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.split(orig_steps, num_segs[i])
            
            warp = np.concatenate(np.random.permutation(np.array(splits, dtype=object))).astype(np.int64)
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return ret
