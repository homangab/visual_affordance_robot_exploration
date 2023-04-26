import os
import torch
import torch.nn.functional as F
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager
from torch.nn.parallel import DistributedDataParallel

from . import AutoEncoderModel, META_ARCH_REGISTRY
from .. import PixelLoss
from ..vq import VQEmbedding
from ..vq.vq_embedding import DVQEmbedding
from ...solver import build_lr_scheduler
from ...solver.build import build_optimizer

from affordance.utils.image import tensor2im



import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# Some basic setup:
# Setup detectron2 logger

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

model = build_model(cfg)
model.train(False)

nc = 3
ndf = 32

torch.autograd.set_detect_anomaly(True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 64 x 64
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print(input.shape)
        return self.main(input)



@META_ARCH_REGISTRY.register()
class VQVAEModel(AutoEncoderModel):
    """
    ref impl: https://github.com/ritheshkumar95/pytorch-vqvae
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_codebook_ema = cfg.MODEL.CODEBOOK.EMA

        if cfg.MODEL.CODEBOOK.NUM == 1:
            self.codebook = VQEmbedding(cfg.MODEL.CODEBOOK.SIZE, cfg.MODEL.CODEBOOK.DIM, self.use_codebook_ema)
        else:
            self.codebook = DVQEmbedding(cfg.MODEL.CODEBOOK.NUM, cfg.MODEL.CODEBOOK.SIZE, cfg.MODEL.CODEBOOK.DIM,
                                         self.use_codebook_ema)

        if self.use_codebook_ema:
            self._set_requires_grad(self.codebook.parameters(), False)

        self.pixel_loss = PixelLoss(cfg)  # TODO move it to ae
        self.beta = cfg.MODEL.CODEBOOK.BETA

        self.discriminator = Discriminator().to('cuda')
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.loss = nn.BCELoss()


        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.codebook.train(mode)
        return self

    def wrap_parallel(self, device_ids, broadcast_buffers):
        super().wrap_parallel(device_ids, broadcast_buffers)
        if not self.use_codebook_ema:
            self.codebook = DistributedDataParallel(self.codebook, device_ids=device_ids,
                                                    broadcast_buffers=broadcast_buffers)

    def _generator_parameters(self):
        params = super()._generator_parameters()
        if not self.use_codebook_ema:
            params += list(self.codebook.parameters())
        return params

    def forward(self, data, mode='inference'):
        return super().forward(data, mode)

    def compute_generator_loss(self, x,x1):
        loss_dict, discriminator_loss, x, x_tilde = self.compute_supervised_loss(x, x1 return_x=True)

        return loss_dict, discriminator_loss

    def compute_supervised_loss(self, x, x1, return_x=False):
        loss_dict = {}

        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook(z_e_x, "st")
        x_recon = self.generator(z_q_x_st)


        # Reconstruction loss
        loss_dict['loss_reconstruction'] = self.pixel_loss(x_recon, x1)


        if return_x:
            return loss_dict,  x, x_recon
        else:
            return loss_dict





    def encode(self, x):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            z_e_x = self.encoder(x.view(b * t, c, h, w))
            latents = self.codebook(z_e_x)  # b * t, h,  w
            return latents.view(b, t, *latents.size()[1:])
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.forward(latents, mode="emb").permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        x_tilde = self.generator(z_q_x)
        return x_tilde

    def configure_optimizers_and_checkpointers(self):
        o, c = super().configure_optimizers_and_checkpointers()

        if not self.use_codebook_ema:
            optimizer_c = build_optimizer(self.codebook, self.cfg, suffix="_G")
            scheduler_c = build_lr_scheduler(self.cfg, optimizer_c)
            o += [
                {"optimizer": optimizer_c, "scheduler": scheduler_c, "type": "generator"},
            ]

        PathManager.mkdirs(os.path.join(self.cfg.OUTPUT_DIR, 'netC'))
        c += [
            {"checkpointer": Checkpointer(self.codebook, os.path.join(self.cfg.OUTPUT_DIR, 'netC')),
             "pretrained": self.cfg.MODEL.CODEBOOK.WEIGHTS, },
        ]

        return o, c
