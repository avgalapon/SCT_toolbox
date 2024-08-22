#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train_class.py
@Time    :   2024/07/16 11:25:18
@Author  :   AVGalapon 
@Contact :   a.v.galapon@umcg.nl
@License :   (C)Copyright 2022-2023, Arthur Galapon
@Desc    :   None
'''


import torch
import random
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Train:
    def __init__(self,config, model, device, loss, train_dataloader, valid_dataloader, plot=False):
        self.config = config
        self.model = model
        self.device = device
        self.loss = loss
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.plot = plot

    def train_DCNN(self):
        model = self.model['DCNN']
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.lambda_decay, momentum=0.9, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=DecayLR(self.config.epoch, 0, self.config.decay_epoch).step)
        
        train_loss, val_loss = {}, {}
        best_loss = np.inf

        for epoch in range(self.config.epoch):
            model.train()
            train_loss[epoch] = self.run_epoch_DCNN(model, self.loss, optimizer, epoch)
            val_loss[epoch] = self.validate_DCNN(model, self.loss, epoch)

            lr_scheduler.step()
            
            best_loss = self.save_best_model_DCNN(model, epoch, val_loss[epoch], best_loss)
            self.save_losses('DCNN', train_loss, val_loss)            
            
        print("Training complete!")            
    
    def run_epoch_DCNN(self, model, loss_fn, optimizer, epoch):
        losses = []
        plot_count = 0
        max_plots = 20
        
        pbar = tqdm(self.train_dataloader, desc="Training")
        for data in pbar:
            real_X, real_Y = data['A'].to(self.device), data['B'].to(self.device)

            optimizer.zero_grad()
            fake_Y = model(real_X)
            loss = loss_fn(real_Y, fake_Y)
            loss += self.compute_l1_norm(model, 0.001)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
            if self.plot and plot_count < max_plots and random.random() < 0.3:
                plot_count += 1
                self.plot_images(real_X, real_Y, fake_Y, save_path=f"{self.config.weights_path}/epoch_{epoch}_training_{plot_count}.png")
            
            running_mean = np.mean(losses)
            pbar.set_description(f"Training Loss: {running_mean}")

        return running_mean
    
    def validate_DCNN(self, model, loss_fn, epoch):
        model.eval()
        losses = []

        with torch.no_grad():
            plot_count = 0
            max_plots = 20
            pbar_val = tqdm(self.valid_dataloader, desc="Validation")
            for data in pbar_val:
                val_X, val_Y = data['A'].to(self.device), data['B'].to(self.device)
                fake_Y = model(val_X)
                loss = loss_fn(val_Y, fake_Y)
                losses.append(loss.item())
                
                if self.plot and plot_count < max_plots and random.random() < 0.3:
                        plot_count += 1
                        self.plot_images(val_X, val_Y, fake_Y, save_path=f"{self.config.weights_path}/epoch_{epoch}_validation_{plot_count}.png")
                        
                running_mean = np.mean(losses)
                pbar_val.set_description(f"Validation Loss: {running_mean}")

        return running_mean

    def save_best_model_DCNN(self, model, epoch, current_loss, best_loss):
        if current_loss < best_loss and epoch > 1:
            best_loss = current_loss
            torch.save(model.state_dict(), f"{self.config.weights_path}/DCNN_weights_best.pth")
            with open(f"{self.config.weights_path}/best_loss_epoch.txt", "w") as f:
                f.write(f"Best loss epoch: {epoch} | loss: {best_loss}")
        return best_loss

    def train_cycleGAN(self):
        model_A2B, model_B2A = self.model['cycleGAN_A2B'], self.model['cycleGAN_B2A']
        model_DA, model_DB = self.model['cycleGAN_DA'], self.model['cycleGAN_DB']
        loss_cycleGAN, loss_MAE = self.loss

        optimizer_G = torch.optim.Adam(itertools.chain(model_A2B.parameters(), model_B2A.parameters()), lr=self.learning_rate, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(itertools.chain(model_DA.parameters(), model_DB.parameters()), lr=self.learning_rate, betas=(0.5, 0.999))
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=DecayLR(self.epoch, 0, self.decay_epoch).step)
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=DecayLR(self.epoch, 0, self.decay_epoch).step)

        train_loss, val_loss = {}, {}
        best_loss = np.inf

        for epoch in range(self.config.epoch):
            model_A2B.train()
            model_B2A.train()
            train_loss[epoch] = self.run_epoch_cycleGAN(model_A2B, model_B2A, model_DA, model_DB, loss_cycleGAN, optimizer_G, optimizer_D, epoch)
            val_loss[epoch] = self.validate_cycleGAN(model_A2B, loss_MAE, epoch)

            lr_scheduler_G.step()
            lr_scheduler_D.step()
            best_loss = self.save_best_model_cycleGAN(model_A2B, model_B2A, model_DA, model_DB, epoch, val_loss[epoch], best_loss)

        self.save_losses('cycleGAN', train_loss, val_loss)
    
    def run_epoch_cycleGAN(self, model_A2B, model_B2A, model_DA, model_DB, loss_cycleGAN, optimizer_G, optimizer_D, epoch):
        loss_G, loss_DA, loss_DB = [], [], []
        plot_count = 0
        max_plots = 20

        for data in tqdm(self.train_dataloader, desc="Training"):
            real_X, real_Y = data['A'].to(self.device), data['B'].to(self.device)

            fake_X = model_B2A(real_Y)
            fake_Y = model_A2B(real_X)
            rec_X = model_B2A(fake_Y)
            rec_Y = model_A2B(fake_X)
            idt_X = model_B2A(real_X)
            idt_Y = model_A2B(real_Y)

            pred_real_X, pred_real_Y = model_DA(real_X), model_DB(real_Y)
            pred_fake_X, pred_fake_Y = model_DA(fake_X.detach()), model_DB(fake_Y.detach())

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            lossG, lossDA, lossDB = loss_cycleGAN(real_X, real_Y, fake_X, fake_Y, rec_X, rec_Y, idt_X, idt_Y, pred_real_X, pred_real_Y, pred_fake_X, pred_fake_Y)
            lossG.backward()
            optimizer_G.step()
            (lossDA + lossDB).backward()
            optimizer_D.step()

            loss_G.append(lossG.item())
            loss_DA.append(lossDA.item())
            loss_DB.append(lossDB.item())
            
            if self.plot and plot_count < max_plots and random.random() < 0.3:
                plot_count += 1
                self.plot_images(real_X, real_Y, fake_Y, save_path=f"{self.config.weights_path}/epoch_{epoch}_training_{plot_count}.png")

        return [np.mean(loss_G), np.mean(loss_DA), np.mean(loss_DB)]

    def validate_cycleGAN(self, model_A2B, loss_MAE, epoch):
        model_A2B.eval()
        losses = []
        plot_count = 0
        max_plots = 20

        with torch.no_grad():
            for data in tqdm(self.valid_dataloader, desc="Validation"):
                val_X, val_Y = data['A'].to(self.device), data['B'].to(self.device)
                fake_Y = model_A2B(val_X)
                loss = loss_MAE(val_Y, fake_Y)
                losses.append(loss.item())
                
                if self.plot and plot_count < max_plots and random.random() < 0.3:
                        plot_count += 1
                        self.plot_images(val_X, val_Y, fake_Y, save_path=f"{self.config.weights_path}/epoch_{epoch}_validation_{plot_count}.png")

        return np.mean(losses)

    def save_best_model_cycleGAN(self, model_A2B, model_B2A, model_DA, model_DB, epoch, current_loss, best_loss):
        if current_loss < best_loss and epoch > 1:
            best_loss = current_loss
            torch.save(model_A2B.state_dict(), f"{self.config.weights_path}/cycleGAN_A2B_weights_best.pth")
            torch.save(model_B2A.state_dict(), f"{self.config.weights_path}/cycleGAN_B2A_weights_best.pth")
            torch.save(model_DA.state_dict(), f"{self.config.weights_path}/cycleGAN_DA_weights_best.pth")
            torch.save(model_DB.state_dict(), f"{self.config.weights_path}/cycleGAN_DB_weights_best.pth")
            with open(f"{self.config.weights_path}/best_loss_epoch.txt", "w") as f:
                f.write(f"Best loss epoch: {epoch} | loss: {best_loss}")
        return best_loss

    def train_cGAN(self):
        model_G, model_D = self.model['cGAN_G'], self.model['cGAN_D']
        loss_laplacian, loss_cGAN, loss_MAE = self.loss

        optimizer_G = torch.optim.Adam(model_G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=DecayLR(self.epoch, 0, self.decay_epoch).step)
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=DecayLR(self.epoch, 0, self.decay_epoch).step)

        train_loss, val_loss = {}, {}
        best_loss = np.inf

        for epoch in range(self.config.epoch):
            model_G.train()
            model_D.train()
            train_loss[epoch] = self.run_epoch_cGAN(model_G, model_D, loss_laplacian, loss_cGAN, optimizer_G, optimizer_D, epoch)
            val_loss[epoch] = self.validate_cGAN(model_G, loss_MAE, epoch)

            lr_scheduler_G.step()
            lr_scheduler_D.step()
            best_loss = self.save_best_model_cGAN(model_G, model_D, epoch, val_loss[epoch], best_loss)

        self.save_losses('cGAN', train_loss, val_loss)
    
    def run_epoch_cGAN(self, model_G, model_D, loss_laplacian, loss_cGAN, optimizer_G, optimizer_D, epoch):
        loss_G, loss_D = [], []
        plot_count = 0
        max_plots = 20

        for data in tqdm(self.train_dataloader, desc="Training"):
            real_X, real_Y = data['A'].to(self.device), data['B'].to(self.device)
            fake_Y, std_Y = model_G(real_X)

            optimizer_D.zero_grad()
            lossG, lossD = loss_cGAN(real_X, fake_Y, model_D(real_Y), model_D(fake_Y))
            lossD.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            total_loss_G = loss_laplacian(real_Y, fake_Y, std_Y) + lossG
            total_loss_G.backward()
            optimizer_G.step()

            loss_G.append(total_loss_G.item())
            loss_D.append(lossD.item())
            
            if self.plot and plot_count < max_plots and random.random() < 0.3:
                plot_count += 1
                self.plot_images(real_X, real_Y, fake_Y, save_path=f"{self.config.weights_path}/epoch_{epoch}_training_{plot_count}.png")

        return [np.mean(loss_G), np.mean(loss_D)]
    
    def validate_cGAN(self, model_G, loss_MAE, epoch):
        model_G.eval()
        losses = []

        with torch.no_grad():
            plot_count = 0
            max_plots = 20
            for data in tqdm(self.valid_dataloader, desc="Validation"):
                val_X, val_Y = data['A'].to(self.device), data['B'].to(self.device)
                fake_Y, _ = model_G(val_X)
                loss = loss_MAE(val_Y, fake_Y)
                losses.append(loss.item())
                
                if self.plot and plot_count < max_plots and random.random() < 0.3:
                    plot_count += 1
                    self.plot_images(val_X, val_Y, fake_Y, save_path=f"{self.config.weights_path}/epoch_{epoch}_validation_{plot_count}.png")

        return np.mean(losses)

    def save_best_model_cGAN(self, model_G, model_D, epoch, current_loss, best_loss):
        if current_loss < best_loss and epoch > 1:
            best_loss = current_loss
            torch.save(model_G.state_dict(), f"{self.config.weights_path}/cGAN_G_weights_best.pth")
            torch.save(model_D.state_dict(), f"{self.config.weights_path}/cGAN_D_weights_best.pth")
            with open(f"{self.config.weights_path}/best_loss_epoch.txt", "w") as f:
                f.write(f"Best loss epoch: {epoch} | loss: {best_loss}")
        return best_loss

    def save_losses(self, model_name, train_loss, val_loss):
        np.save(f"{self.config.weights_path}/{model_name}_train_loss.npy", train_loss)
        np.save(f"{self.config.weights_path}/{model_name}_val_loss.npy", val_loss)
    
    def plot_train_val_curve(self,model_name):
        train_loss = np.load(f"{self.config.weights_path}/{model_name}_train_loss.npy", allow_pickle=True)
        val_loss = np.load(f"{self.config.weights_path}/{model_name}_val_loss.npy", allow_pickle=True)
        
        epochs = sorted(train_loss.keys())
        train_loss_values = [train_loss[epoch] for epoch in epochs]
        val_loss_values = [val_loss[epoch] for epoch in epochs]
        
        # Plot the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss_values, label='Training Loss')
        plt.plot(epochs, val_loss_values, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Loss')
        plt.legend()
        plt.savefig(f"{self.config.weights_path}/{model_name}_loss_curve.png")
        plt.close()
        
    def compute_l1_norm(self, model, lambda1=0.5):
        """Compute L1 regularization."""
        l1_regularization = torch.tensor(0).to(self.device, dtype=torch.float)

        for param in model.parameters():
            l1_regularization += torch.norm(param, 1)

        return lambda1 * l1_regularization
    
    def plot_images(self, real_X, real_Y, fake_Y, save_path=None):
        # Transfer tensors from GPU to CPU and convert to numpy arrays
        real_X = real_X.detach().cpu().numpy()
        real_Y = real_Y.detach().cpu().numpy()
        fake_Y = fake_Y.detach().cpu().numpy()
        
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        diff = np.abs(real_Y - fake_Y)
        
        im0 = ax[0].imshow(real_X[0, 0, :, :], cmap='gray')
        ax[0].set_title('Real X')
        ax[0].axis('off')
        
        im1 = ax[1].imshow(real_Y[0, 0, :, :], cmap='gray')
        ax[1].set_title('Real Y')
        ax[1].axis('off')
        
        im2 = ax[2].imshow(fake_Y[0, 0, :, :], cmap='gray')
        ax[2].set_title('Fake Y')
        ax[2].axis('off')
        
        im3 = ax[3].imshow(diff[0, 0, :, :], cmap='RdBu_r')
        ax[3].set_title('Difference')
        
        # Create a divider for the existing axis to append an axis for the colorbar
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add colorbar for the difference plot
        cbar = fig.colorbar(im3, cax=cax)
        cbar.set_label('Difference Intensity')
        
        # Save the figure if a save_path is provided
        if save_path:
            plt.savefig(save_path)
            
        plt.close()

class DecayLR:
    def __init__(self, epochs, offset, decay_epochs):
        epoch_flag = epochs - decay_epochs
        assert (epoch_flag > 0), "Decay must start before the training session ends!"
        self.epochs = epochs
        self.offset = offset
        self.decay_epochs = decay_epochs

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (
                self.epochs - self.decay_epochs)
