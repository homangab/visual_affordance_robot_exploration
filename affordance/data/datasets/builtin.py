

import os

from .latents import register_latents


import os

from affordance.data import DatasetCatalog, MetadataCatalog
from affordance.utils.image import get_video_paths, get_image_paths

def load_vids(root, phase, load_images):
    """
    Returns list of dicts
    each dict contains:
        video_path: path to the video
        image_paths: sorted list of image paths
    """
    if load_images:
        return get_image_paths(os.path.join(root, phase))
    return get_video_paths(os.path.join(root, phase))


def register_final(name, root,phase, load_images):
    DatasetCatalog.register(name, lambda: load_vids(root, phase, load_images))
    MetadataCatalog.get(name).set(root=root)

def register_all_final(root="data/"):
    SPLITS = [
        ("final_train", "train", True),
        ("final_test_seq", "train", False),
    ]
    for name, phase, load_images in SPLITS:
        register_final(name, os.path.join(root), phase,load_images)



register_all_final()
register_latents("latent_train", "experiments/vqvae/inference/final_test_seq/")
register_latents("latent_test", "experiments/vqvae/inference/final_test_seq/")
