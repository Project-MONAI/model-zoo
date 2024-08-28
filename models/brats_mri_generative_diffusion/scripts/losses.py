# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import torch
from monai.losses.adversarial_loss import PatchAdversarialLoss

intensity_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")

adv_weight = 0.1
perceptual_weight = 0.1
# kl_weight: important hyper-parameter.
#     If too large, decoder cannot recon good results from latent space.
#     If too small, latent space will not be regularized enough for the diffusion model
kl_weight = 1e-7


def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def generator_loss(gen_images, real_images, z_mu, z_sigma, disc_net, loss_perceptual):
    recons_loss = intensity_loss(gen_images, real_images)
    kl_loss = compute_kl_loss(z_mu, z_sigma)
    p_loss = loss_perceptual(gen_images.float(), real_images.float())
    loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

    logits_fake = disc_net(gen_images)[-1]
    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    loss_g = loss_g + adv_weight * generator_loss

    return loss_g


def discriminator_loss(gen_images, real_images, disc_net):
    logits_fake = disc_net(gen_images.contiguous().detach())[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    logits_real = disc_net(real_images.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
    loss_d = adv_weight * discriminator_loss
    return loss_d
