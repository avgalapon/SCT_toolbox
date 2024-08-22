"""
Copyright (c) 2018 Maria Francesca Spadea.
- All Rights Reserved -

Unauthorized copying/distributing/editing/using/selling of this file (also partial), via any medium, is strictly prohibited.

The code is proprietary and confidential.

The software is just for research purpose, it is not intended for clinical application or for use in which the failure of the software
could lead to death, personal injury, or severe physical or environmental damage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import random


class GlobalMAE(nn.Module):
    """
    Global MAE
    """

    def __init__(self):
        super(GlobalMAE, self).__init__()

    def forward(self, gt, comp):
        return torch.mean(torch.abs(gt-comp))

class MaskedMAE(nn.Module):
    """
    Masked MAE
    """

    def __init__(self):
        super(MaskedMAE, self).__init__()

    def forward(self, gt, comp, skin, weight=1.0):
        return weight * torch.sum(torch.abs(skin * (gt - comp)))/torch.sum(skin)

class MaskedMAEPlusSSIM(nn.Module):
    """
    Masked MAE plus SSIM
    """

    def __init__(self):
        super(MaskedMAEPlusSSIM, self).__init__()

    def forward(self, gt, comp, skin, ssim_weight):
        maskedMAE = torch.sum(torch.abs(skin * (gt - comp)))/torch.sum(skin)

        comp[skin==0]=-1000

        ssim_losser = pytorch_ssim.SSIM(window_size = 11)
        ssim_loss = 1.0 - ssim_losser(gt, comp)
        total_ssim_loss = ssim_loss * ssim_weight

        print("MAE = %.2f -- SSIM = %.2f -- TOTAL SSIM = %.2f" % (float(maskedMAE), float(ssim_loss), float(total_ssim_loss)))

        return maskedMAE + total_ssim_loss        

class LaplacianLoss(nn.Module):
    def __init__(self,lambda_Laplace=100):
        super(LaplacianLoss, self).__init__()
        self.lambda_Laplace = lambda_Laplace

    def forward(self, fake_data, real_data, std_output):        
        laplace_loss = torch.mean(std_output + torch.abs(real_data - fake_data)*torch.exp(-std_output))
        return self.lambda_Laplace*laplace_loss #
    
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.register_buffer('target_real_label', torch.tensor(target_real_label))
        self.register_buffer('target_fake_label', torch.tensor(target_fake_label))

    def forward(self, prediction, target_is_real):
        target_label = self.target_real_label if target_is_real else self.target_fake_label
        target_tensor = target_label.expand_as(prediction).to(prediction.device)
        return self.loss(prediction, target_tensor)

class ReplayBuffer:
    def __init__(self, max_size=100):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class CycleGANLoss(nn.Module):
    def __init__(self, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5):
        super(CycleGANLoss, self).__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity

        self.gan_loss = GANLoss(target_real_label=1.0, target_fake_label=0.0)
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def forward(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B, idt_A, idt_B,
                pred_real_A, pred_real_B, pred_fake_A, pred_fake_B):
        # Use the replay buffer for fake_A and fake_B
        fake_A_buffered = self.fake_A_buffer.push_and_pop(fake_A)
        fake_B_buffered = self.fake_B_buffer.push_and_pop(fake_B)

        # Adversarial loss for G_A
        loss_G_A = self.gan_loss(pred_fake_B, target_is_real=True)
        # Adversarial loss for G_B
        loss_G_B = self.gan_loss(pred_fake_A, target_is_real=True)

        # Cycle consistency loss
        loss_cycle_A = self.cycle_loss(rec_A, real_A) * self.lambda_A
        loss_cycle_B = self.cycle_loss(rec_B, real_B) * self.lambda_B

        # Identity loss
        loss_idt_A = self.identity_loss(idt_A, real_A) * self.lambda_A * self.lambda_identity
        loss_idt_B = self.identity_loss(idt_B, real_B) * self.lambda_B * self.lambda_identity

        # Total loss for generators
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        # Adversarial loss for D_A
        loss_D_A_real = self.gan_loss(pred_real_A, target_is_real=True)
        loss_D_A_fake = self.gan_loss(fake_A_buffered, target_is_real=False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # Adversarial loss for D_B
        loss_D_B_real = self.gan_loss(pred_real_B, target_is_real=True)
        loss_D_B_fake = self.gan_loss(fake_B_buffered, target_is_real=False)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        return loss_G, loss_D_A, loss_D_B


class cGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(cGANLoss, self).__init__()
        self.gan_loss = GANLoss(target_real_label=target_real_label, target_fake_label=target_fake_label)
        self.fake_buffer = ReplayBuffer()
    def forward(self, real_data, fake_data, pred_real, pred_fake):
        """
        Compute the cGAN loss for both the generator and the discriminator.

        Parameters:
        - real_data: The real data used for the discriminator's real prediction.
        - fake_data: The data generated by the generator used for the discriminator's fake prediction.
        - pred_real: The discriminator's prediction for the real data.
        - pred_fake: The discriminator's prediction for the fake data generated by the generator.

        Returns:
        - loss_G: The loss for the generator.
        - loss_D: The loss for the discriminator.
        """
        
        # Adversarial loss for the generator
        loss_G = self.gan_loss(pred_fake, target_is_real=True)

        # Adversarial loss for the discriminator
        fake_A_buffered = self.fake_buffer.push_and_pop(fake_data)
        loss_D_real = self.gan_loss(pred_real, target_is_real=True)
        loss_D_fake = self.gan_loss(fake_A_buffered, target_is_real=False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return loss_G, loss_D


def define_loss(config):
    modeltype = config.model_type
    
    if modeltype == 'DCNN':
        return GlobalMAE()
    elif modeltype == 'cycleGAN':
        return CycleGANLoss(), GlobalMAE()
    elif modeltype == 'cGAN':
        return LaplacianLoss(), cGANLoss(), GlobalMAE()
    else:
        raise ValueError("Model type not recognized")


def compute_l1_norm(model, device, lambda1=0.5):
        
    """ This function computes l1 regularization."""
    l1_regularization = torch.tensor(0).to(device, dtype=torch.float)

    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)

    return lambda1 * l1_regularization