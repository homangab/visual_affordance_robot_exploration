import argparse
import os
import numpy as np
import torch
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
from affordance.config import get_cfg
from affordance.engine import default_setup
from affordance.modeling.meta_arch import build_model
from affordance.utils.image import save_image, get_image_paths, read_image

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    return cfg


def load_frame(frames_dir, scale_to_zeroone=True):
    """
    Args:
        frames_dir: path to folder with initial images
        scale_to_zeroone: scale the image by 255 or not
    """
    img_paths = [x['image_path'] for x in get_image_paths(frames_dir, use_cache=False)]
    img = [np.ascontiguousarray(read_image(img_path)).transpose(2, 0, 1) for img_path in img_paths]
    img = np.stack(img, axis=0).astype('float32')
    if scale_to_zeroone:
        img /= 255.
    return img


def save_frame(img, output_dir):
    """
    Save img
    Args:
        img: shape [B, H, W, C]
        frames_dir: save new frame under frames_dir/sample/
    """
    PathManager.mkdirs(output_dir)
    for frame_idx in range(len(img)):
        frame_path = os.path.join(output_dir, f'{frame_idx}.png')
        save_image(img[frame_idx], frame_path)

def save_video(video, output_dir):
    """
    Save video
    Args:
        video: shape [T, H, W, C]
        video_dir: save video under video_dir/sample/
    """
    PathManager.mkdirs(output_dir)
    for frame_idx in range(len(video)):
        frame_path = os.path.join(output_dir, f'{frame_idx}.png')
        save_image(video[frame_idx], frame_path)

@torch.no_grad()
def sample_frames(args):
    # load config
    cfg = setup(args)
    cfg.TEST.EVALUATORS = "VTSampler"
    cfg.TEST.NUM_SAMPLES = 1

    # load affordancetransformer
    affordancetransformer = build_model(cfg)
    Checkpointer(affordancetransformer.model).resume_or_load(cfg.MODEL.GENERATOR.WEIGHTS, resume=False)
    affordancetransformer.eval()

    # load vqvae
    vq_cfg = get_cfg()
    vq_cfg.merge_from_file(cfg.TEST.VT_SAMPLER.VQ_VAE.CFG)
    vqvae = build_model(vq_cfg)
    Checkpointer(vqvae.encoder).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.ENCODER_WEIGHTS, resume=False)
    Checkpointer(vqvae.generator).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.GENERATOR_WEIGHTS, resume=False)
    Checkpointer(vqvae.codebook).resume_or_load(cfg.TEST.VT_SAMPLER.VQ_VAE.CODEBOOK_WEIGHTS, resume=False)
    vqvae.eval()

    # load data
    scale_to_zeroone = vq_cfg.INPUT.SCALE_TO_ZEROONE
    image = load_frame(args.frames_dir)[:1]  
    assert image.shape == (1, 3, 128, 128) ## set to image dimension

    # sample
    latent = vqvae([{'image_sequence': image}])[0]['latent'] 
    print(f"Converted to latent codes.")
    _, nc, h, w = latent.shape
    new = latent.new_zeros(2, nc, h, w) ## change first to 2,4 etc. depending on num future frames
    new[:1] = latent
    samples = affordancetransformer([{'image_sequence': new}])[0]['samples']  # list of samples
    print(f"Sampled new frame.")
    sample = samples[0].squeeze(0)  
    if sample.dim() == 4:
        sample = sample.transpose(0, 1)  
    sample = vqvae.decode(sample) 
    sample = vqvae.back_normalizer(sample)  
    print(sample.shape)
    if scale_to_zeroone:
        sample = sample * 255
    sample.clamp_(0.0, 255.0)
    sample = sample.permute(0, 2, 3, 1).contiguous() 
    sample = sample.detach().cpu().numpy().astype(np.uint8)
    save_frame(sample, cfg.OUTPUT_DIR)
    print(f"Saved new frame.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample new frame given initial frame")
    parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--frames-dir", required=True, help="path to folder with initial frames")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    sample_frames(args)
