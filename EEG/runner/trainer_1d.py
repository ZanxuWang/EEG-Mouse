# import random
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import wandb
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from diffusers.optimization import get_cosine_schedule_with_warmup
# from mmengine import Config
# from ..registry import EEGDiffMR, EEGDiffDR
# from torchvision.transforms import Compose, Normalize
# from ..pipeline import DDIMPipeline1D

# @EEGDiffMR.register_module()
# class EEGDiffTrainner1D:
#     def __init__(self,
#                  trainner_config: Config,
#                  unet: Config,
#                  noise_scheduler: Config,
#                  optimizer: Config,
#                  train_dataset: Config,
#                  val_dataset: Config):
#         self.config = trainner_config
#         self.unet = EEGDiffMR.build(unet)
#         self.initial_unet()
#         self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
#         self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
#         self.train_dataset = EEGDiffDR.build(train_dataset)
#         self.val_dataset = EEGDiffDR.build(val_dataset)

#         # Define a custom normalization function
#         def normalize_1d(tensor):
#             return (tensor - 0.5) / 0.5  # Equivalent to Normalize(mean=[0.5], std=[0.5])

#         # Apply normalization manually
#         self.train_dataset.transform = normalize_1d
#         self.val_dataset.transform = normalize_1d
        
#         self.train_dataloader = DataLoader(
#             self.train_dataset, 
#             batch_size=self.config.train_batch_size, 
#             shuffle=True
#         )
#         self.val_dataloader = DataLoader(
#             self.val_dataset, 
#             batch_size=self.config.eval_batch_size, 
#             shuffle=True
#         )
        
#         self.optimizer = torch.optim.Adam(
#             self.unet.parameters(), 
#             lr=optimizer.learning_rate
#         )
#         self.lr_scheduler = get_cosine_schedule_with_warmup(
#             optimizer=self.optimizer,
#             num_warmup_steps=self.config.lr_warmup_steps,
#             num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
#         )

#     def initial_unet(self):
#         self.unet.to(self.config.device)
#         if self.config.u_net_weight_path is not None:
#             self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
#             print(f"Load u_net weight from {self.config.u_net_weight_path}")
#         else:
#             print("No u_net weight path is provided, using random weights")

#     def evaluate(self, batch, epoch):
#         original_signal = batch[0].to(self.config.device)
#         father_path = f"{self.config.output_dir}/{epoch}"
#         if not os.path.exists(father_path):
#             os.makedirs(father_path)
            
#         # Generate predicted signal
#         result = self.pipeline(
#             original_signal,
#             self.config.prediction_point,
#             batch_size=len(original_signal),
#             num_inference_steps=self.config.num_train_timesteps,
#         )
        
#         # Extract the predicted signal
#         predicted_signal = result.images
        
#         # Normalize original signal for comparison
#         original_signal = (original_signal / 2 + 0.5).clamp(0, 1)
#         original_signal_np = original_signal.cpu().numpy()
        
#         # Calculate difference between original and predicted
#         diff = np.sum((original_signal_np - predicted_signal.cpu().numpy()) ** 2)
#         wandb.log({'signal diff': diff})

#         # Plot a random sample for visualization
#         index = np.random.randint(0, len(original_signal) - 1)
        
#         # Plot predicted signal
#         plt.figure(figsize=(15, 5))
#         plt.plot(predicted_signal[index, 0, :])
#         plt.title('Predicted Signal')
#         plt.savefig(f"{father_path}/predicted_signal.png")
#         plt.close()
        
#         # Plot original signal
#         plt.figure(figsize=(15, 5))
#         plt.plot(original_signal_np[index, 0, :])
#         plt.title('Original Signal')
#         plt.savefig(f"{father_path}/original_signal.png")
#         plt.close()
        
#         # Plot comparison
#         plt.figure(figsize=(15, 5))
#         plt.plot(original_signal_np[index, 0, :], label='Original')
#         plt.plot(predicted_signal[index, 0, :], label='Predicted')
#         plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
#         plt.legend()
#         plt.title('Original vs Predicted Signal')
#         plt.savefig(f"{father_path}/comparison.png")
#         plt.close()
        
#         # Save model weights
#         torch.save(self.unet.state_dict(), f"{father_path}/unet.pth")

#     def train_single_batch(self, batch, epoch):
#         clean_signals = batch[0].to(self.config.device)
        
#         # Generate noise
#         noise = torch.randn(clean_signals.shape).to(clean_signals.device)
        
#         # Sample a random timestep for each example
#         timesteps = torch.randint(
#             0, 
#             self.noise_scheduler.num_train_timesteps,
#             (clean_signals.shape[0],),
#             device=clean_signals.device
#         ).long()
        
#         # Add noise to signals
#         signals_with_noise = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
#         # Keep the first part of the signal unchanged (conditioning)
#         signals_with_noise[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
#         # Predict the noise residual
#         noise_pred = self.unet(signals_with_noise, timesteps, return_dict=False)[0]
        
#         # Compute loss only on the predicted part
#         loss = F.mse_loss(
#             noise_pred[:, :, self.config.prediction_point:],
#             noise[:, :, self.config.prediction_point:]
#         )
        
#         # Backpropagate
#         loss.backward()
#         self.optimizer.step()
#         self.lr_scheduler.step()
#         self.optimizer.zero_grad()
        
#         # Log results
#         logs = {
#             "epoch": (epoch // len(self.train_dataloader)),
#             "iteration": epoch,
#             "mse loss": loss.detach().item(),
#             "lr": self.lr_scheduler.get_last_lr()[0]
#         }
#         print(", ".join([key + ": " + str(round(value, 5)) for key, value in logs.items()]))
#         wandb.log(logs)
        
#         return loss.item()

#     def train(self):
#         number_iteration = 0
#         for epoch in range(self.config.num_epochs):
#             for iteration, batch in enumerate(self.train_dataloader):
#                 self.train_single_batch(batch, number_iteration)
#                 number_iteration += 1
                
#                 # Evaluate periodically
#                 if (number_iteration >= self.config.eval_begin and 
#                     number_iteration % self.config.eval_interval == 0):
#                     # Get a random batch from validation set
#                     index = random.randint(1, len(self.val_dataloader) - 1)
#                     for eval_iteration, eval_batch in enumerate(self.val_dataloader):
#                         if eval_iteration == index:
#                             self.evaluate(eval_batch, number_iteration)
#                             break


import random
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
from mmengine import Config
from ..registry import EEGDiffMR, EEGDiffDR
from ..pipeline import DDIMPipeline1D

@EEGDiffMR.register_module()
class EEGDiffTrainner1D:
    def __init__(self,
                 trainner_config: Config,
                 unet: Config,
                 noise_scheduler: Config,
                 optimizer: Config,
                 train_dataset: Config,
                 val_dataset: Config):
        self.config = trainner_config
        self.unet = EEGDiffMR.build(unet)
        self.initial_unet()
        self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
        self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
        # Define a custom normalization function
        def normalize_1d(tensor):
            return (tensor - 0.5) / 0.5  # Equivalent to Normalize(mean=[0.5], std=[0.5])
        
        self.train_dataset = EEGDiffDR.build(train_dataset)
        self.val_dataset = EEGDiffDR.build(val_dataset)
        self.train_dataset.transform = normalize_1d
        self.val_dataset.transform = normalize_1d
        
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.train_batch_size, 
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.eval_batch_size, 
            shuffle=True
        )
        
        self.optimizer = torch.optim.Adam(
            self.unet.parameters(), 
            lr=optimizer.learning_rate
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print(f"Load u_net weight from {self.config.u_net_weight_path}")
        else:
            print("No u_net weight path is provided, using random weights")

    def evaluate(self, batch, epoch):
        original_signal = batch[0].to(self.config.device)
        father_path = f"{self.config.output_dir}/{epoch}"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
            
        # Generate predicted signal
        result = self.pipeline(
            original_signal,
            self.config.prediction_point,
            batch_size=len(original_signal),
            num_inference_steps=self.config.num_train_timesteps,
        )
        
        # Extract the predicted signal
        predicted_signal = result.images
        
        # Normalize original signal for comparison
        original_signal = (original_signal / 2 + 0.5).clamp(0, 1)
        
        # Move tensors to CPU for processing with numpy
        predicted_signal_np = predicted_signal.cpu().numpy()
        original_signal_np = original_signal.cpu().numpy()
        
        # Calculate difference between original and predicted
        diff = np.sum((original_signal_np - predicted_signal_np) ** 2)
        wandb.log({'signal diff': diff})

        # Plot a random sample for visualization
        index = np.random.randint(0, len(original_signal) - 1)
        
        # Plot predicted signal
        plt.figure(figsize=(15, 5))
        plt.plot(predicted_signal_np[index, 0, :])
        plt.title('Predicted Signal')
        plt.savefig(f"{father_path}/predicted_signal.png")
        plt.close()
        
        # Plot original signal
        plt.figure(figsize=(15, 5))
        plt.plot(original_signal_np[index, 0, :])
        plt.title('Original Signal')
        plt.savefig(f"{father_path}/original_signal.png")
        plt.close()
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        plt.plot(original_signal_np[index, 0, :], label='Original')
        plt.plot(predicted_signal_np[index, 0, :], label='Predicted')
        plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
        plt.legend()
        plt.title('Original vs Predicted Signal')
        plt.savefig(f"{father_path}/comparison.png")
        plt.close()
        
        # Save model weights
        torch.save(self.unet.state_dict(), f"{father_path}/unet.pth")

    def train_single_batch(self, batch, epoch):
        clean_signals = batch[0].to(self.config.device)
        
        # Generate noise
        noise = torch.randn(clean_signals.shape).to(clean_signals.device)
        
        # Sample a random timestep for each example
        timesteps = torch.randint(
            0, 
            self.noise_scheduler.config.num_train_timesteps,  # Use config.num_train_timesteps to avoid warning
            (clean_signals.shape[0],),
            device=clean_signals.device
        ).long()
        
        # Add noise to signals
        signals_with_noise = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
        # Keep the first part of the signal unchanged (conditioning)
        signals_with_noise[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
        # Predict the noise residual
        noise_pred = self.unet(signals_with_noise, timesteps).sample
        
        # Compute loss only on the predicted part
        loss = F.mse_loss(
            noise_pred[:, :, self.config.prediction_point:],
            noise[:, :, self.config.prediction_point:]
        )
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        # Log results
        logs = {
            "epoch": (epoch // len(self.train_dataloader)),
            "iteration": epoch,
            "mse loss": loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0]
        }
        print(", ".join([key + ": " + str(round(value, 5)) for key, value in logs.items()]))
        wandb.log(logs)
        
        return loss.item()

    def train(self):
        number_iteration = 0
        for epoch in range(self.config.num_epochs):
            for iteration, batch in enumerate(self.train_dataloader):
                self.train_single_batch(batch, number_iteration)
                number_iteration += 1
                
                # Evaluate periodically
                if (number_iteration >= self.config.eval_begin and 
                    number_iteration % self.config.eval_interval == 0):
                    # Get a random batch from validation set
                    index = random.randint(0, len(self.val_dataloader) - 1)
                    for eval_iteration, eval_batch in enumerate(self.val_dataloader):
                        if eval_iteration == index:
                            self.evaluate(eval_batch, number_iteration)
                            break