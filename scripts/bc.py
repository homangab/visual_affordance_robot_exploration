import argparse
import os
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
from affordance.config import get_cfg
from affordance.engine import default_setup
from affordance.modeling.meta_arch import build_model
from affordance.utils.image import save_image, get_image_paths, read_image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import glob
from PIL import Image
from replay_buffer import ReplayBuffer
from policy import DiscreteStochasticGoalPolicy, FeatureExtract 



transform = transforms.Compose([  # [1]
    transforms.Resize(240),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3925, 0.3798, 0.3587],
                            std=[0.1723, 0.1577, 0.1554])
])


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


@torch.no_grad()
def sample_frames(args,image,only_init=False):
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
    if image is None:
        image = load_frame(args.frames_dir)[:1]  
    assert image.shape == (1, 3, 128, 128)


    # sample
    latent = vqvae([{'image_sequence': image}])[0]['latent'] 
    _, nc, h, w = latent.shape
    new = latent.new_zeros(2, nc, h, w)
    new[:1] = latent
    if only_init:
        return new

    samples = affordancetransformer([{'image_sequence': new}])[0]['samples'] 
    sample = samples[0].squeeze(0)  
    if sample.dim() == 4:
        sample = sample.transpose(0, 1)  
    
    init_goal_latent = sample 
    sample = vqvae.decode(sample) 
    sample = vqvae.back_normalizer(sample)  
    if scale_to_zeroone:
        sample = sample * 255
    sample.clamp_(0.0, 255.0)
    sample = sample.permute(0, 2, 3, 1).contiguous() 
    sample = sample.detach().cpu().numpy().astype(np.uint8)
    save_frame(sample, cfg.OUTPUT_DIR)

    return init_goal_latent,sample


def trajectory(args,policy):
    
    if args.realsense:
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        align_to = rs.stream.depth
        align = rs.align(align_to)
        aligned_frames = align.process(frames)
        color_array=np.array(color_frame.get_data())
        image = color_array
    else:
        image = None

    ## choose goal 
    init_goal_latent,_ = sample_frames(args,image) ## sample latent goal
    init_goal_latent = init_goal_latent.float()
    current_latent = init_goal_latent[0]
    goal_latent = init_goal_latent[1]

    states = []
    actions = []
    goals = []
    for t in range(args.horizon):        
        current_latent = torch.flatten(current_latent)
        goal_latent = torch.flatten(goal_latent)
        feature_extract = True

        if feature_extract:
            #image = Image.open("./example/1.png")
            _,goal_image = sample_frames(args,image)
            image_numpy = goal_image[0]

            if len(image_numpy.shape) == 2:
                image_numpy = np.expand_dims(image_numpy, axis=2)
            if image_numpy.shape[2] == 1:
                image_numpy = np.repeat(image_numpy, 3, 2)
            image_pil = Image.fromarray(image_numpy)
            img_transformed = transform(image_pil)
            img_transformed = img_transformed[None, :]
            goal_img = img_transformed
            current_latent, goal_latent = feature_extractor(img_transformed,goal_img)
        
        action = policy(current_latent,goal_latent)

        if args.realsense:
            fa.goto_pose(action[0:6]) 
            if action[6] > 0.5:
                fa.open_gripper()
            else:
                fa.close_gripper()

        states.append(current_latent.cpu().detach().numpy())
        actions.append(action.cpu().detach().numpy())
        goals.append(goal_latent.cpu().detach().numpy())

        ## get next observation
        if args.realsense:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            align_to = rs.stream.depth
            align = rs.align(align_to)
            aligned_frames = align.process(frames)
            color_array=np.array(color_frame.get_data())
            image = color_array
        else:
            image = None
        
        ## get latent of current frame
        init_latent = sample_frames(args,image,only_init=True).float()
        current_latent = init_latent[0]

    ## open gripper; move it back to home position
    if args.realsense:
        fa.open_gripper()
        fa.reset_pose() 
    
    ### set last image as goal
    for i in range(len(goals)):
        goals[i] = torch.flatten(current_latent).cpu().detach().numpy()

    return np.stack(states), np.array(actions), np.stack(goals)


def train(args,policy,replay_buffer,policy_optimizer):

    for epochs in range(args.epoch):
        for episodes in range(args.explore_ep):
            states, actions, goal_latent = trajectory(args,policy)
            replay_buffer.add_trajectory(states, actions, goal_latent)
        
        for train_step in range(args.steps_per_epoch):
            policy_step(args,policy,replay_buffer,policy_optimizer)
            print("policy_step")


def loss_fn(observations, goals, actions, horizons, weights,policy):
    obs_dtype = torch.float32
    action_dtype = torch.float32

    observations_torch = torch.tensor(observations, dtype=obs_dtype)
    goals_torch = torch.tensor(goals, dtype=obs_dtype)
    actions_torch = torch.tensor(actions, dtype=action_dtype)
    horizons_torch = torch.tensor(horizons, dtype=obs_dtype)

    conditional_nll = policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)
    nll = conditional_nll

    return torch.mean(nll)

def policy_step(args,policy,replay_buffer,policy_optimizer):

    policy_optimizer.zero_grad()
    
    for _ in range(args.policy_step):
        observations, actions, goals, _, horizons, weights = replay_buffer.sample_batch(args.batch_size)
        loss = loss_fn(observations, goals, actions, horizons, weights,policy)
        loss.backward()

    policy_optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self Supervised Exploration and Policy Learning")
    parser.add_argument("--config-file", required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--frames-dir", required=False, default='example',help="path to folder with initial frames")
    parser.add_argument("--horizon", type=int, default=100, help="episode horizon")
    parser.add_argument("--epoch", type=int, default=100, help="train epochs")
    parser.add_argument("--steps_per_epoch", type=int,default=20, help="training steps per epoch")
    parser.add_argument("--explore_ep", type=int,default=20, help="exploration episodes per epoch")
    parser.add_argument("--batch_size", type=int,default=64, help="batch size for training")
    parser.add_argument("--policy_step", type=int,default=10, help="batch size for training")
    parser.add_argument('--realsense', action='store_true', help="set True if there is a camera attached")
    parser.add_argument('--feature_extract', action='store_true', help="If pretrained resnet needs to extract im features")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    if args.realsense: ## lazyimport 
        import pyrealsense2 as rs
        from frankapy import FrankaArm
        pipe = rs.pipeline()
        profile = pipe.start()
        fa = FrankaArm()

    policy = DiscreteStochasticGoalPolicy(feature_extract=args.feature_extract)
    feature_extractor = FeatureExtract()
    replay_buffer = ReplayBuffer(max_trajectory_length=args.horizon,buffer_size=300)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    train(args,policy,replay_buffer,policy_optimizer)
