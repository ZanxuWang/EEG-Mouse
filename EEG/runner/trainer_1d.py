# # import random
# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import wandb
# # import torch
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader
# # from diffusers.optimization import get_cosine_schedule_with_warmup
# # from mmengine import Config
# # from ..registry import EEGDiffMR, EEGDiffDR
# # from torchvision.transforms import Compose, Normalize
# # from ..pipeline import DDIMPipeline1D

# # @EEGDiffMR.register_module()
# # class EEGDiffTrainner1D:
# #     def __init__(self,
# #                  trainner_config: Config,
# #                  unet: Config,
# #                  noise_scheduler: Config,
# #                  optimizer: Config,
# #                  train_dataset: Config,
# #                  val_dataset: Config):
# #         self.config = trainner_config
# #         self.unet = EEGDiffMR.build(unet)
# #         self.initial_unet()
# #         self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
# #         self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
# #         self.train_dataset = EEGDiffDR.build(train_dataset)
# #         self.val_dataset = EEGDiffDR.build(val_dataset)

# #         # Define a custom normalization function
# #         def normalize_1d(tensor):
# #             return (tensor - 0.5) / 0.5  # Equivalent to Normalize(mean=[0.5], std=[0.5])

# #         # Apply normalization manually
# #         self.train_dataset.transform = normalize_1d
# #         self.val_dataset.transform = normalize_1d
        
# #         self.train_dataloader = DataLoader(
# #             self.train_dataset, 
# #             batch_size=self.config.train_batch_size, 
# #             shuffle=True
# #         )
# #         self.val_dataloader = DataLoader(
# #             self.val_dataset, 
# #             batch_size=self.config.eval_batch_size, 
# #             shuffle=True
# #         )
        
# #         self.optimizer = torch.optim.Adam(
# #             self.unet.parameters(), 
# #             lr=optimizer.learning_rate
# #         )
# #         self.lr_scheduler = get_cosine_schedule_with_warmup(
# #             optimizer=self.optimizer,
# #             num_warmup_steps=self.config.lr_warmup_steps,
# #             num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
# #         )

# #     def initial_unet(self):
# #         self.unet.to(self.config.device)
# #         if self.config.u_net_weight_path is not None:
# #             self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
# #             print(f"Load u_net weight from {self.config.u_net_weight_path}")
# #         else:
# #             print("No u_net weight path is provided, using random weights")

# #     def evaluate(self, batch, epoch):
# #         original_signal = batch[0].to(self.config.device)
# #         father_path = f"{self.config.output_dir}/{epoch}"
# #         if not os.path.exists(father_path):
# #             os.makedirs(father_path)
            
# #         # Generate predicted signal
# #         result = self.pipeline(
# #             original_signal,
# #             self.config.prediction_point,
# #             batch_size=len(original_signal),
# #             num_inference_steps=self.config.num_train_timesteps,
# #         )
        
# #         # Extract the predicted signal
# #         predicted_signal = result.images
        
# #         # Normalize original signal for comparison
# #         original_signal = (original_signal / 2 + 0.5).clamp(0, 1)
# #         original_signal_np = original_signal.cpu().numpy()
        
# #         # Calculate difference between original and predicted
# #         diff = np.sum((original_signal_np - predicted_signal.cpu().numpy()) ** 2)
# #         wandb.log({'signal diff': diff})

# #         # Plot a random sample for visualization
# #         index = np.random.randint(0, len(original_signal) - 1)
        
# #         # Plot predicted signal
# #         plt.figure(figsize=(15, 5))
# #         plt.plot(predicted_signal[index, 0, :])
# #         plt.title('Predicted Signal')
# #         plt.savefig(f"{father_path}/predicted_signal.png")
# #         plt.close()
        
# #         # Plot original signal
# #         plt.figure(figsize=(15, 5))
# #         plt.plot(original_signal_np[index, 0, :])
# #         plt.title('Original Signal')
# #         plt.savefig(f"{father_path}/original_signal.png")
# #         plt.close()
        
# #         # Plot comparison
# #         plt.figure(figsize=(15, 5))
# #         plt.plot(original_signal_np[index, 0, :], label='Original')
# #         plt.plot(predicted_signal[index, 0, :], label='Predicted')
# #         plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
# #         plt.legend()
# #         plt.title('Original vs Predicted Signal')
# #         plt.savefig(f"{father_path}/comparison.png")
# #         plt.close()
        
# #         # Save model weights
# #         torch.save(self.unet.state_dict(), f"{father_path}/unet.pth")

# #     def train_single_batch(self, batch, epoch):
# #         clean_signals = batch[0].to(self.config.device)
        
# #         # Generate noise
# #         noise = torch.randn(clean_signals.shape).to(clean_signals.device)
        
# #         # Sample a random timestep for each example
# #         timesteps = torch.randint(
# #             0, 
# #             self.noise_scheduler.num_train_timesteps,
# #             (clean_signals.shape[0],),
# #             device=clean_signals.device
# #         ).long()
        
# #         # Add noise to signals
# #         signals_with_noise = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
# #         # Keep the first part of the signal unchanged (conditioning)
# #         signals_with_noise[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
# #         # Predict the noise residual
# #         noise_pred = self.unet(signals_with_noise, timesteps, return_dict=False)[0]
        
# #         # Compute loss only on the predicted part
# #         loss = F.mse_loss(
# #             noise_pred[:, :, self.config.prediction_point:],
# #             noise[:, :, self.config.prediction_point:]
# #         )
        
# #         # Backpropagate
# #         loss.backward()
# #         self.optimizer.step()
# #         self.lr_scheduler.step()
# #         self.optimizer.zero_grad()
        
# #         # Log results
# #         logs = {
# #             "epoch": (epoch // len(self.train_dataloader)),
# #             "iteration": epoch,
# #             "mse loss": loss.detach().item(),
# #             "lr": self.lr_scheduler.get_last_lr()[0]
# #         }
# #         print(", ".join([key + ": " + str(round(value, 5)) for key, value in logs.items()]))
# #         wandb.log(logs)
        
# #         return loss.item()

# #     def train(self):
# #         number_iteration = 0
# #         for epoch in range(self.config.num_epochs):
# #             for iteration, batch in enumerate(self.train_dataloader):
# #                 self.train_single_batch(batch, number_iteration)
# #                 number_iteration += 1
                
# #                 # Evaluate periodically
# #                 if (number_iteration >= self.config.eval_begin and 
# #                     number_iteration % self.config.eval_interval == 0):
# #                     # Get a random batch from validation set
# #                     index = random.randint(1, len(self.val_dataloader) - 1)
# #                     for eval_iteration, eval_batch in enumerate(self.val_dataloader):
# #                         if eval_iteration == index:
# #                             self.evaluate(eval_batch, number_iteration)
# #                             break


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
        
#         # Define a custom normalization function
#         def normalize_1d(tensor):
#             return (tensor - 0.5) / 0.5  # Equivalent to Normalize(mean=[0.5], std=[0.5])
        
#         self.train_dataset = EEGDiffDR.build(train_dataset)
#         self.val_dataset = EEGDiffDR.build(val_dataset)
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

#     # def evaluate(self, batch, epoch):
#     #     original_signal = batch[0].to(self.config.device)
#     #     father_path = f"{self.config.output_dir}/{epoch}"
#     #     if not os.path.exists(father_path):
#     #         os.makedirs(father_path)
            
#     #     # Generate predicted signal
#     #     result = self.pipeline(
#     #         original_signal,
#     #         self.config.prediction_point,
#     #         batch_size=len(original_signal),
#     #         num_inference_steps=self.config.num_train_timesteps,
#     #     )
        
#     #     # Extract the predicted signal
#     #     predicted_signal = result.images
        
#     #     # Normalize original signal for comparison
#     #     original_signal = (original_signal / 2 + 0.5).clamp(0, 1)
        
#     #     # Move tensors to CPU for processing with numpy
#     #     predicted_signal_np = predicted_signal.cpu().numpy()
#     #     original_signal_np = original_signal.cpu().numpy()
        
#     #     # Calculate difference between original and predicted
#     #     diff = np.sum((original_signal_np - predicted_signal_np) ** 2)
#     #     wandb.log({'signal diff': diff})

#     #     # Plot a random sample for visualization
#     #     index = np.random.randint(0, len(original_signal) - 1)
        
#     #     # Plot predicted signal
#     #     plt.figure(figsize=(15, 5))
#     #     plt.plot(predicted_signal_np[index, 0, :])
#     #     plt.title('Predicted Signal')
#     #     plt.savefig(f"{father_path}/predicted_signal.png")
#     #     plt.close()
        
#     #     # Plot original signal
#     #     plt.figure(figsize=(15, 5))
#     #     plt.plot(original_signal_np[index, 0, :])
#     #     plt.title('Original Signal')
#     #     plt.savefig(f"{father_path}/original_signal.png")
#     #     plt.close()
        
#     #     # Plot comparison
#     #     plt.figure(figsize=(15, 5))
#     #     plt.plot(original_signal_np[index, 0, :], label='Original')
#     #     plt.plot(predicted_signal_np[index, 0, :], label='Predicted')
#     #     plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
#     #     plt.legend()
#     #     plt.title('Original vs Predicted Signal')
#     #     plt.savefig(f"{father_path}/comparison.png")
#     #     plt.close()
        
#     #     # Save model weights
#     #     torch.save(self.unet.state_dict(), f"{father_path}/unet.pth")

#     def evaluate(self, batch, epoch):
#         original_signal = batch[0].to(self.config.device)
#         father_path = f"{self.config.output_dir}/{epoch}"
#         if not os.path.exists(father_path):
#             os.makedirs(father_path)
        
#         # Create a directory specifically for sample visualizations
#         samples_dir = os.path.join(father_path, "samples")
#         if not os.path.exists(samples_dir):
#             os.makedirs(samples_dir)
        
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
        
#         # Move tensors to CPU for processing with numpy
#         predicted_signal_np = predicted_signal.cpu().numpy()
#         original_signal_np = original_signal.cpu().numpy()
        
#         # Calculate difference between original and predicted
#         diff = np.sum((original_signal_np - predicted_signal_np) ** 2)
#         wandb.log({'signal diff': diff})
        
#         # Calculate per-sample MSE for the predicted part
#         sample_mses = []
#         for i in range(len(original_signal)):
#             pred_part_orig = original_signal_np[i, 0, self.config.prediction_point:]
#             pred_part_pred = predicted_signal_np[i, 0, self.config.prediction_point:]
#             mse = np.mean((pred_part_pred - pred_part_orig) ** 2)
#             sample_mses.append(mse)
        
#         # Log average MSE
#         wandb.log({'avg_prediction_mse': np.mean(sample_mses)})
        
#         # Determine number of samples to visualize - either 10 or the batch size if smaller
#         num_samples = min(10, len(original_signal))
        
#         # Pick samples with diverse prediction quality (some good, some bad predictions)
#         # Sort by MSE and sample evenly
#         sorted_indices = np.argsort(sample_mses)
#         step = max(1, len(sorted_indices) // num_samples)
#         selected_indices = sorted_indices[::step][:num_samples]
        
#         # Create individual plots for each selected sample
#         for i, idx in enumerate(selected_indices):
#             # Plot predicted signal
#             plt.figure(figsize=(15, 5))
#             plt.plot(predicted_signal_np[idx, 0, :])
#             plt.title(f'Sample {i+1}: Predicted Signal (MSE: {sample_mses[idx]:.4f})')
#             plt.savefig(f"{samples_dir}/sample_{i+1}_predicted.png")
#             plt.close()
            
#             # Plot original signal
#             plt.figure(figsize=(15, 5))
#             plt.plot(original_signal_np[idx, 0, :])
#             plt.title(f'Sample {i+1}: Original Signal')
#             plt.savefig(f"{samples_dir}/sample_{i+1}_original.png")
#             plt.close()
            
#             # Plot comparison
#             plt.figure(figsize=(15, 5))
#             plt.plot(original_signal_np[idx, 0, :], label='Original')
#             plt.plot(predicted_signal_np[idx, 0, :], label='Predicted')
#             plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
#             plt.legend()
#             plt.title(f'Sample {i+1}: Original vs Predicted Signal (MSE: {sample_mses[idx]:.4f})')
#             plt.savefig(f"{samples_dir}/sample_{i+1}_comparison.png")
#             plt.close()
        
#         # Create a multi-sample comparison plot (all samples in one figure)
#         fig, axs = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples), sharex=True)
#         for i, idx in enumerate(selected_indices):
#             if num_samples == 1:
#                 ax = axs  # If only one sample, axs is not a list
#             else:
#                 ax = axs[i]
                
#             ax.plot(original_signal_np[idx, 0, :], 'b-', label='Original')
#             ax.plot(predicted_signal_np[idx, 0, :], 'g-', label='Predicted')
#             ax.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
#             ax.set_title(f'Sample {i+1} (MSE: {sample_mses[idx]:.4f})')
#             ax.legend()
        
#         plt.tight_layout()
#         plt.savefig(f"{father_path}/multi_sample_comparison.png")
#         plt.close()
        
#         # Create a summary visualization of prediction quality distribution
#         plt.figure(figsize=(10, 6))
#         plt.hist(sample_mses, bins=20)
#         plt.title(f'Distribution of MSE Across Samples (Avg: {np.mean(sample_mses):.4f})')
#         plt.xlabel('MSE')
#         plt.ylabel('Count')
#         plt.savefig(f"{father_path}/mse_distribution.png")
#         plt.close()
        
#         # Save model weights
#         torch.save(self.unet.state_dict(), f"{father_path}/unet.pth")

#     # def train_single_batch(self, batch, epoch):
#     #     clean_signals = batch[0].to(self.config.device)
        
#     #     # Generate noise
#     #     noise = torch.randn(clean_signals.shape).to(clean_signals.device)
        
#     #     # Sample a random timestep for each example
#     #     timesteps = torch.randint(
#     #         0, 
#     #         self.noise_scheduler.config.num_train_timesteps,  # Use config.num_train_timesteps to avoid warning
#     #         (clean_signals.shape[0],),
#     #         device=clean_signals.device
#     #     ).long()
        
#     #     # Add noise to signals
#     #     signals_with_noise = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
#     #     # Keep the first part of the signal unchanged (conditioning)
#     #     signals_with_noise[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]

#     def train_single_batch(self, batch, epoch):
#         clean_signals = batch[0].to(self.config.device)
        
#         # Update here: Scale down the noise
#         noise_scale = 1  # Experiment with values between 0.01-0.5
#         noise = noise_scale * torch.randn(clean_signals.shape).to(clean_signals.device)
        
#         # Rest of the method remains the same
#         timesteps = torch.randint(
#             0, 
#             self.noise_scheduler.config.num_train_timesteps,
#             (clean_signals.shape[0],),
#             device=clean_signals.device
#         ).long()
        
#         # Add scaled noise to signals
#         signals_with_noise = self.noise_scheduler.add_noise(clean_signals, noise, timesteps)
        
#         # Keep the first part of the signal unchanged (conditioning)
#         signals_with_noise[:, :, :self.config.prediction_point] = clean_signals[:, :, :self.config.prediction_point]
        
#         # Predict the noise residual
#         noise_pred = self.unet(signals_with_noise, timesteps).sample
        
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
#                     index = random.randint(0, len(self.val_dataloader) - 1)
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
import pandas as pd
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
        
        # Load labels for seizure/non-seizure classification
        try:
            train_labels_path = getattr(self.config, 'train_labels_path', 'C:/Users/chris/EEG-DIF/data/train_labels.csv')
            test_labels_path = getattr(self.config, 'test_labels_path', 'C:/Users/chris/EEG-DIF/data/test_labels.csv')
            
            self.train_labels = pd.read_csv(train_labels_path, header=None).values.flatten()
            self.test_labels = pd.read_csv(test_labels_path, header=None).values.flatten()
            
            print(f"Loaded {len(self.train_labels)} training labels and {len(self.test_labels)} testing labels")
        except Exception as e:
            print(f"Error loading labels: {e}")
            print("Will continue without seizure/non-seizure classification")
            self.train_labels = None
            self.test_labels = None

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print(f"Load u_net weight from {self.config.u_net_weight_path}")
        else:
            print("No u_net weight path is provided, using random weights")

    def evaluate(self, batch, epoch, is_training_set=False):
        """
        Enhanced evaluation method with seizure/non-seizure comparison
        
        Args:
            batch: The batch of data to evaluate
            epoch: Current epoch/iteration
            is_training_set: Whether this batch is from training set (True) or test set (False)
        """
        original_signal = batch[0].to(self.config.device)
        father_path = f"{self.config.output_dir}/{epoch}"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        
        # Create directories for sample visualizations and dataset-specific plots
        samples_dir = os.path.join(father_path, "samples")
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)
        
        dataset_dir = os.path.join(father_path, "train_test_comparison")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
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
        
        # Calculate per-sample MSE for the predicted part
        sample_mses = []
        for i in range(len(original_signal)):
            pred_part_orig = original_signal_np[i, 0, self.config.prediction_point:]
            pred_part_pred = predicted_signal_np[i, 0, self.config.prediction_point:]
            mse = np.mean((pred_part_pred - pred_part_orig) ** 2)
            sample_mses.append(mse)
        
        # Log average MSE
        wandb.log({'avg_prediction_mse': np.mean(sample_mses)})
        
        # Get batch labels if available
        labels = None
        if is_training_set and self.train_labels is not None:
            # This is more complex since we don't know exactly which samples from training set are in this batch
            # For now, we'll just check if we have enough labels for the whole dataset
            if len(self.train_labels) >= len(self.train_dataset):
                # Just use the first N labels as a simple approach
                # This won't match the exact samples but gives us something to work with
                labels = self.train_labels[:len(original_signal)]
                dataset_name = "training"
        elif not is_training_set and self.test_labels is not None:
            # Same simplification for test set
            if len(self.test_labels) >= len(self.val_dataset):
                labels = self.test_labels[:len(original_signal)]
                dataset_name = "testing"
        
        # Create seizure/non-seizure visualizations if labels are available
        if labels is not None:
            # Separate seizure and non-seizure samples
            seizure_indices = [i for i, label in enumerate(labels) if label == 1]
            non_seizure_indices = [i for i, label in enumerate(labels) if label == 0]
            
            # Skip if we don't have both classes in this batch
            if len(seizure_indices) > 0 and len(non_seizure_indices) > 0:
                # Calculate average MSE for seizure and non-seizure samples
                seizure_mse = np.mean([sample_mses[i] for i in seizure_indices])
                non_seizure_mse = np.mean([sample_mses[i] for i in non_seizure_indices])
                
                wandb.log({
                    f'{dataset_name}_seizure_mse': seizure_mse,
                    f'{dataset_name}_non_seizure_mse': non_seizure_mse
                })
                
                pred_start = self.config.prediction_point
                
                # 1. Plot all seizure samples in one figure
                plt.figure(figsize=(15, 8))
                for idx in seizure_indices:
                    # Only plot the predicted part for clarity
                    plt.plot(range(pred_start, len(predicted_signal_np[idx, 0, :])), 
                             predicted_signal_np[idx, 0, pred_start:], 'r-', alpha=0.3)
                
                # Plot average of seizure predictions
                avg_seizure_pred = np.mean([predicted_signal_np[idx, 0, pred_start:] for idx in seizure_indices], axis=0)
                plt.plot(range(pred_start, len(predicted_signal_np[0, 0, :])), 
                         avg_seizure_pred, 'r-', linewidth=3, label='Avg Seizure Prediction')
                
                # Plot average of seizure originals
                avg_seizure_orig = np.mean([original_signal_np[idx, 0, pred_start:] for idx in seizure_indices], axis=0)
                plt.plot(range(pred_start, len(original_signal_np[0, 0, :])), 
                         avg_seizure_orig, 'b-', linewidth=3, label='Avg Seizure Original')
                
                plt.title(f'{dataset_name.capitalize()} Seizure Predictions (MSE: {seizure_mse:.4f})')
                plt.legend()
                plt.savefig(f"{dataset_dir}/{dataset_name}_seizure_predictions.png")
                plt.close()
                
                # 2. Plot all non-seizure samples in one figure
                plt.figure(figsize=(15, 8))
                for idx in non_seizure_indices:
                    # Only plot the predicted part for clarity
                    plt.plot(range(pred_start, len(predicted_signal_np[idx, 0, :])), 
                             predicted_signal_np[idx, 0, pred_start:], 'g-', alpha=0.3)
                
                # Plot average of non-seizure predictions
                avg_non_seizure_pred = np.mean([predicted_signal_np[idx, 0, pred_start:] for idx in non_seizure_indices], axis=0)
                plt.plot(range(pred_start, len(predicted_signal_np[0, 0, :])), 
                         avg_non_seizure_pred, 'g-', linewidth=3, label='Avg Non-Seizure Prediction')
                
                # Plot average of non-seizure originals
                avg_non_seizure_orig = np.mean([original_signal_np[idx, 0, pred_start:] for idx in non_seizure_indices], axis=0)
                plt.plot(range(pred_start, len(original_signal_np[0, 0, :])), 
                         avg_non_seizure_orig, 'b-', linewidth=3, label='Avg Non-Seizure Original')
                
                plt.title(f'{dataset_name.capitalize()} Non-Seizure Predictions (MSE: {non_seizure_mse:.4f})')
                plt.legend()
                plt.savefig(f"{dataset_dir}/{dataset_name}_non_seizure_predictions.png")
                plt.close()
                
                # 3. Combined plot with average seizure vs non-seizure
                plt.figure(figsize=(15, 8))
                
                plt.plot(range(pred_start, len(predicted_signal_np[0, 0, :])), 
                         avg_seizure_pred, 'r-', linewidth=2, label='Avg Seizure Prediction')
                plt.plot(range(pred_start, len(original_signal_np[0, 0, :])), 
                         avg_seizure_orig, 'r--', linewidth=2, label='Avg Seizure Original')
                
                plt.plot(range(pred_start, len(predicted_signal_np[0, 0, :])), 
                         avg_non_seizure_pred, 'g-', linewidth=2, label='Avg Non-Seizure Prediction')
                plt.plot(range(pred_start, len(original_signal_np[0, 0, :])), 
                         avg_non_seizure_orig, 'g--', linewidth=2, label='Avg Non-Seizure Original')
                
                plt.title(f'{dataset_name.capitalize()} Average Seizure vs Non-Seizure Comparison')
                plt.legend()
                plt.savefig(f"{dataset_dir}/{dataset_name}_avg_seizure_vs_non_seizure.png")
                plt.close()
                
                # Create a bar chart for seizure vs non-seizure MSE
                plt.figure(figsize=(10, 6))
                plt.bar(['Seizure', 'Non-Seizure'], [seizure_mse, non_seizure_mse])
                plt.title(f'{dataset_name.capitalize()} MSE Comparison: Seizure vs Non-Seizure')
                plt.ylabel('Mean Squared Error')
                plt.savefig(f"{dataset_dir}/{dataset_name}_seizure_vs_non_seizure_mse.png")
                plt.close()
            else:
                print(f"Warning: Batch does not contain both seizure and non-seizure samples")
        
        # Determine number of samples to visualize - either 10 or the batch size if smaller
        num_samples = min(10, len(original_signal))
        
        # Pick samples with diverse prediction quality (some good, some bad predictions)
        # Sort by MSE and sample evenly
        sorted_indices = np.argsort(sample_mses)
        step = max(1, len(sorted_indices) // num_samples)
        selected_indices = sorted_indices[::step][:num_samples]
        
        # Create individual plots for each selected sample
        for i, idx in enumerate(selected_indices):
            # Plot predicted signal
            plt.figure(figsize=(15, 5))
            plt.plot(predicted_signal_np[idx, 0, :])
            
            # Add seizure/non-seizure label if available
            label_str = ""
            if labels is not None and idx < len(labels):
                label_str = " (Seizure)" if labels[idx] == 1 else " (Non-Seizure)"
                
            plt.title(f'Sample {i+1}: Predicted Signal (MSE: {sample_mses[idx]:.4f}){label_str}')
            plt.savefig(f"{samples_dir}/sample_{i+1}_predicted.png")
            plt.close()
            
            # Plot original signal
            plt.figure(figsize=(15, 5))
            plt.plot(original_signal_np[idx, 0, :])
            plt.title(f'Sample {i+1}: Original Signal{label_str}')
            plt.savefig(f"{samples_dir}/sample_{i+1}_original.png")
            plt.close()
            
            # Plot comparison
            plt.figure(figsize=(15, 5))
            plt.plot(original_signal_np[idx, 0, :], label='Original')
            plt.plot(predicted_signal_np[idx, 0, :], label='Predicted')
            plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
            plt.legend()
            plt.title(f'Sample {i+1}: Original vs Predicted Signal (MSE: {sample_mses[idx]:.4f}){label_str}')
            plt.savefig(f"{samples_dir}/sample_{i+1}_comparison.png")
            plt.close()
        
        # Create a multi-sample comparison plot (all samples in one figure)
        fig, axs = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples), sharex=True)
        for i, idx in enumerate(selected_indices):
            if num_samples == 1:
                ax = axs  # If only one sample, axs is not a list
            else:
                ax = axs[i]
                
            ax.plot(original_signal_np[idx, 0, :], 'b-', label='Original')
            ax.plot(predicted_signal_np[idx, 0, :], 'g-', label='Predicted')
            ax.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
            
            # Add seizure/non-seizure label if available
            label_str = ""
            if labels is not None and idx < len(labels):
                label_str = " (Seizure)" if labels[idx] == 1 else " (Non-Seizure)"
                
            ax.set_title(f'Sample {i+1} (MSE: {sample_mses[idx]:.4f}){label_str}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{father_path}/multi_sample_comparison.png")
        plt.close()
        
        # Create a summary visualization of prediction quality distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sample_mses, bins=20)
        plt.title(f'Distribution of MSE Across Samples (Avg: {np.mean(sample_mses):.4f})')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        plt.savefig(f"{father_path}/mse_distribution.png")
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

    def evaluate_full_datasets(self, epoch):
        """
        Evaluate on full training and testing datasets to get comprehensive comparison
        between seizure and non-seizure performance
        """
        print("Evaluating on full training and testing datasets...")
        
        # Create directory for full evaluation results
        full_eval_dir = f"{self.config.output_dir}/{epoch}/full_evaluation"
        if not os.path.exists(full_eval_dir):
            os.makedirs(full_eval_dir)
        
        # Function to process an entire dataset and create visualizations
        def evaluate_dataset(dataloader, dataset_name, labels=None):
            print(f"Processing entire {dataset_name} dataset...")
            all_originals = []
            all_predictions = []
            all_mses = []
            
            # Process all batches
            for batch_idx, batch in enumerate(dataloader):
                # Get batch data
                original_signal = batch[0].to(self.config.device)
                
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
                batch_predictions = predicted_signal.cpu().numpy()
                batch_originals = original_signal.cpu().numpy()
                
                # Calculate per-sample MSE for the predicted part
                for i in range(len(original_signal)):
                    pred_part_orig = batch_originals[i, 0, self.config.prediction_point:]
                    pred_part_pred = batch_predictions[i, 0, self.config.prediction_point:]
                    mse = np.mean((pred_part_pred - pred_part_orig) ** 2)
                    all_mses.append(mse)
                
                # Store all signals for later analysis
                all_originals.extend([batch_originals[i, 0, :] for i in range(len(batch_originals))])
                all_predictions.extend([batch_predictions[i, 0, :] for i in range(len(batch_predictions))])
                
                print(f"Processed batch {batch_idx+1}/{len(dataloader)} of {dataset_name} dataset")
            
            # Convert to numpy arrays for easier processing
            all_originals = np.array(all_originals)
            all_predictions = np.array(all_predictions)
            all_mses = np.array(all_mses)
            
            # Log overall MSE
            avg_mse = np.mean(all_mses)
            wandb.log({f'{dataset_name}_full_dataset_mse': avg_mse})
            
            # Create MSE distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(all_mses, bins=30)
            plt.title(f'{dataset_name.capitalize()} - Full Dataset MSE Distribution (Avg: {avg_mse:.4f})')
            plt.xlabel('MSE')
            plt.ylabel('Count')
            plt.savefig(f"{full_eval_dir}/{dataset_name}_full_mse_distribution.png")
            plt.close()
            
            # Sample visualization - plot some random samples
            num_samples = min(5, len(all_originals))
            plt.figure(figsize=(15, 4*num_samples))
            
            for i in range(num_samples):
                idx = np.random.randint(0, len(all_originals))
                plt.subplot(num_samples, 1, i+1)
                plt.plot(all_originals[idx], 'b-', label='Original')
                plt.plot(all_predictions[idx], 'g-', label='Predicted')
                plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
                plt.title(f'Sample {idx} (MSE: {all_mses[idx]:.4f})')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{full_eval_dir}/{dataset_name}_sample_comparisons.png")
            plt.close()
            
            # If labels are available, create seizure/non-seizure comparisons
            if labels is not None and len(labels) >= len(all_originals):
                # Limit labels to the number of processed samples
                labels = labels[:len(all_originals)]
                
                # Separate seizure and non-seizure samples
                seizure_indices = [i for i, label in enumerate(labels) if label == 1]
                non_seizure_indices = [i for i, label in enumerate(labels) if label == 0]
                
                if len(seizure_indices) > 0 and len(non_seizure_indices) > 0:
                    # Calculate average MSE for each class
                    seizure_mse = np.mean([all_mses[i] for i in seizure_indices])
                    non_seizure_mse = np.mean([all_mses[i] for i in non_seizure_indices])
                    
                    wandb.log({
                        f'{dataset_name}_full_seizure_mse': seizure_mse,
                        f'{dataset_name}_full_non_seizure_mse': non_seizure_mse
                    })
                    
                    # Create bar chart comparing MSEs
                    plt.figure(figsize=(10, 6))
                    plt.bar(['Seizure', 'Non-Seizure'], [seizure_mse, non_seizure_mse])
                    plt.title(f'{dataset_name.capitalize()} - Full Dataset MSE by Class')
                    plt.ylabel('Mean Squared Error')
                    plt.savefig(f"{full_eval_dir}/{dataset_name}_class_mse_comparison.png")
                    plt.close()
                    
                    # Plot all seizure and non-seizure predictions
                    pred_start = self.config.prediction_point
                    
                    # Seizure samples
                    plt.figure(figsize=(15, 8))
                    
                    # Plot individual seizure samples with low opacity
                    for idx in seizure_indices[:50]:  # Limit to 50 samples for clarity
                        plt.plot(range(pred_start, len(all_predictions[idx])), 
                                 all_predictions[idx][pred_start:], 'r-', alpha=0.1)
                    
                    # Calculate and plot average
                    avg_seizure_orig = np.mean([all_originals[i][pred_start:] for i in seizure_indices], axis=0)
                    avg_seizure_pred = np.mean([all_predictions[i][pred_start:] for i in seizure_indices], axis=0)
                    
                    plt.plot(range(pred_start, len(all_predictions[0])), 
                             avg_seizure_pred, 'r-', linewidth=2, label='Avg Seizure Prediction')
                    plt.plot(range(pred_start, len(all_originals[0])), 
                             avg_seizure_orig, 'b-', linewidth=2, label='Avg Seizure Original')
                    
                    plt.title(f'{dataset_name.capitalize()} - Full Dataset Seizure Predictions (n={len(seizure_indices)})')
                    plt.legend()
                    plt.savefig(f"{full_eval_dir}/{dataset_name}_full_seizure_predictions.png")
                    plt.close()
                    
                    # Non-seizure samples
                    plt.figure(figsize=(15, 8))
                    
                    # Plot individual non-seizure samples with low opacity
                    for idx in non_seizure_indices[:50]:  # Limit to 50 samples for clarity
                        plt.plot(range(pred_start, len(all_predictions[idx])), 
                                 all_predictions[idx][pred_start:], 'g-', alpha=0.1)
                    
                    # Calculate and plot average
                    avg_non_seizure_orig = np.mean([all_originals[i][pred_start:] for i in non_seizure_indices], axis=0)
                    avg_non_seizure_pred = np.mean([all_predictions[i][pred_start:] for i in non_seizure_indices], axis=0)
                    
                    plt.plot(range(pred_start, len(all_predictions[0])), 
                             avg_non_seizure_pred, 'g-', linewidth=2, label='Avg Non-Seizure Prediction')
                    plt.plot(range(pred_start, len(all_originals[0])), 
                             avg_non_seizure_orig, 'b-', linewidth=2, label='Avg Non-Seizure Original')
                    
                    plt.title(f'{dataset_name.capitalize()} - Full Dataset Non-Seizure Predictions (n={len(non_seizure_indices)})')
                    plt.legend()
                    plt.savefig(f"{full_eval_dir}/{dataset_name}_full_non_seizure_predictions.png")
                    plt.close()
                    
                    # Combined comparison plot
                    plt.figure(figsize=(15, 8))
                    
                    plt.plot(range(pred_start, len(all_predictions[0])), 
                             avg_seizure_pred, 'r-', linewidth=2, label='Avg Seizure Prediction')
                    plt.plot(range(pred_start, len(all_originals[0])), 
                             avg_seizure_orig, 'r--', linewidth=2, label='Avg Seizure Original')
                    
                    plt.plot(range(pred_start, len(all_predictions[0])), 
                             avg_non_seizure_pred, 'g-', linewidth=2, label='Avg Non-Seizure Prediction')
                    plt.plot(range(pred_start, len(all_originals[0])), 
                             avg_non_seizure_orig, 'g--', linewidth=2, label='Avg Non-Seizure Original')
                    
                    plt.title(f'{dataset_name.capitalize()} - Full Dataset Seizure vs Non-Seizure Comparison')
                    plt.legend()
                    plt.savefig(f"{full_eval_dir}/{dataset_name}_full_combined_comparison.png")
                    plt.close()
                else:
                    print(f"Warning: {dataset_name} dataset does not contain both seizure and non-seizure samples")
            
            return all_mses
        
        # 1. Evaluate on training dataset
        print("Evaluating on full training dataset...")
        train_mses = evaluate_dataset(self.train_dataloader, "training", self.train_labels)
        
        # 2. Evaluate on testing dataset
        print("Evaluating on full testing dataset...")
        test_mses = evaluate_dataset(self.val_dataloader, "testing", self.test_labels)
        
        # 3. Compare training vs testing performance
        plt.figure(figsize=(10, 6))
        plt.boxplot([train_mses, test_mses], labels=['Training', 'Testing'])
        plt.title('MSE Comparison: Training vs Testing')
        plt.ylabel('Mean Squared Error')
        plt.savefig(f"{full_eval_dir}/train_vs_test_mse_boxplot.png")
        plt.close()
        
        print("Full dataset evaluation completed!")

    def train(self):
        number_iteration = 0
        for epoch in range(self.config.num_epochs):
            for iteration, batch in enumerate(self.train_dataloader):
                self.train_single_batch(batch, number_iteration)
                number_iteration += 1
                
                # Evaluate periodically with enhanced seizure/non-seizure comparison
                if (number_iteration >= self.config.eval_begin and 
                    number_iteration % self.config.eval_interval == 0):
                    # Run evaluation on full datasets
                    self.evaluate_full_datasets(number_iteration)