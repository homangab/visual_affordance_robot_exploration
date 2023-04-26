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

    def compute_generator_loss(self, x):
        loss_dict, discriminator_loss, x, x_tilde = self.compute_supervised_loss(x, return_x=True)

        return loss_dict, discriminator_loss

    def compute_supervised_loss(self, x, return_x=False):
        loss_dict = {}

        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook(z_e_x, "st")
        x_recon = self.generator(z_q_x_st)

        x_recon1 = x_recon.clone()
        x1 = x.clone() 

        masking = True

        fake_data = []
        real_data = []

        if masking:
           # for batch in range(16):
                #print("x.shape",x.shape) #x.shape torch.Size([2, 3, 128, 128])

                #im1 = x.permute(0,2,3,1)[batch].clone()
                #im2 = x_recon.permute(0,2,3,1)[batch].clone()
                #print(im1)
                #print(im1.dtype)
                #im = (im1.cpu().numpy() * 255).astype('uint8')
                #im2 = (im2.cpu().detach().numpy() * 255).astype('uint8')
            #inputs = [{'image':torch.from_numpy(x[batch].clone().cpu().numpy()[::-1, :, :].copy())} for batch in range(16)] 
                #outputs1 = predictor(im)
            #outputs1 = model(inputs)

            for batch in range(16):
                im = (x.clone().permute(0,2,3,1)[batch].cpu().numpy()[:, :, ::-1].copy() * 255).astype('uint8')
                im = tensor2im(x[batch].clone())[:, :, ::-1].copy()
                #print(im)

                outputs1 = predictor(im)



                

                #print('boo')
                if outputs1['instances'].pred_masks.numel() != 0 and (outputs1['instances'].pred_classes == 0).nonzero().numel() != 0:


                    index = (outputs1['instances'].pred_classes == 0).nonzero()[0]




                    mask = outputs1['instances'].pred_masks[index].clone()
                    masked_pred = torch.mul(x_recon[batch].clone(), mask[0].clone().expand(3,128,128)) 
                    masked_inp = torch.mul(x[batch].clone()[0], mask[0].clone().expand(3,128,128)) 
                    masked_pred1 = masked_pred.clone()
                    masked_inp1 = masked_inp.clone()
                    x_recon1[batch,:,:,:] = masked_pred1.clone()
                    x1[batch,:,:,:] = masked_inp1.clone()

                    ## for adversarial loss
                    fake_data.append(x_recon[batch,:,:,:].clone())


                    ### for visualization
                    # pred_classes = outputs1['instances'].pred_classes.cpu().tolist()
                    # class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
                    # pred_class_names = list(map(lambda x: class_names[x], pred_classes))
                    # #print(pred_class_names)

                    # #print((outputs1[batch]["instances"].pred_classes == 0).nonzero())
                    # #print(pred_class_names)
                
                    # #index = (outputs1[batch]["instances"].pred_classes == 0).nonzero()[0]

                    # #print(x)



                    # #print(outputs1)

                    # v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    # out = v.draw_instance_predictions(outputs1['instances'].to("cpu"))
                    # filename = 'detectron_out/detectron_'+ str(batch)+'.png'
                    # cv2.imwrite(filename,out.get_image())

                    # filename1 = 'detectron_out/detectron_init_'+ str(batch)+'.png'

                    # cv2.imwrite(filename1,im)

                else:
                    #real_data.append(x_recon[batch,:,:,:].clone())
                    real_data.append(x[batch,:,:,:].clone())






            loss_dict['loss_reconstruction'] = self.pixel_loss(x_recon1, x1)
        else:
            # Reconstruction loss
            loss_dict['loss_reconstruction'] = self.pixel_loss(x_recon1, x1)





        # Vector quantization objective
        if not self.use_codebook_ema:
            loss_dict['loss_dict'] = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        loss_dict['loss_commitment'] = self.beta * F.mse_loss(z_e_x, z_q_x.detach())
        self.discriminator.zero_grad()




        ### adversarial loss
        #print(len(real_data))
        #print(len(fake_data))

        if len(real_data) == 0 or len(fake_data) == 0:
            print('skipping gan loss')
        else:
            true_labels = torch.ones(len(real_data)).to('cuda')
            fake_labels = torch.zeros(len(fake_data)).to('cuda')
            true_labels1 = torch.ones(len(fake_data)).to('cuda')
            fake_data_tensor = torch.stack(fake_data).clone().to('cuda')
            #real_data = fake_data
            real_data_tensor = torch.stack(real_data).clone().to('cuda')

            #print(true_labels.shape)
            #print(real_data_tensor.shape)


            # Train the discriminator on the true/generated data
            self.discriminator_optimizer.zero_grad()
            true_discriminator_out = self.discriminator(real_data_tensor)
            true_discriminator_loss = self.loss(true_discriminator_out.view(-1), true_labels)

            # add .detach() here think about this
            generator_discriminator_out = self.discriminator(fake_data_tensor.detach())
            generator_discriminator_loss = self.loss(generator_discriminator_out.view(-1), fake_labels)
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            #print('doingnnn')



            generator_discriminator_out = self.discriminator(fake_data_tensor)
            generator_loss = self.loss(generator_discriminator_out.view(-1), true_labels1)

            #print(generator_loss)

            loss_dict['generator_loss'] = generator_loss


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
