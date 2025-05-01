# from mmengine import Config
# from ..registry import EEGDiffMR, EEGDiffDR
# from torchvision.transforms import Compose, Normalize
# from ..pipeline import DDIMPipeline1D
# from torch.utils.data import DataLoader
# from ..dataset.EEG_dataset_1d import EvaluationDataset1D
# import torch
# from diffusers.utils.torch_utils import randn_tensor
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import os

# @EEGDiffMR.register_module()
# class EEGDiffEvaler1D:
#     def __init__(self,
#                  unet: Config,
#                  dataset: Config,
#                  evaler_config: Config,
#                  noise_scheduler: Config):
#         self.config = evaler_config
#         self.unet = EEGDiffMR.build(unet)
#         self.initial_unet()
#         self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
        
#         # Create dataset for evaluation
#         self.dataset = EvaluationDataset1D(
#             csv_path=self.config.csv_path,
#             window_size=self.config.window_size,
#             prediction_point=self.config.prediction_point,
#             step_size=self.config.window_size - self.config.prediction_point
#         )
        
#         # Setup pipeline for inference
#         self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
#         # Calculate batch size
#         self.batch_size = (self.dataset.total_segments) 
        
#         # Create dataloader
#         self.dataloader = DataLoader(
#             dataset=self.dataset, 
#             batch_size=self.batch_size, 
#             shuffle=False
#         )

#     def initial_unet(self):
#         self.unet.to(self.config.device)
#         if self.config.u_net_weight_path is not None:
#             self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
#             print(f"Load u_net weight from {self.config.u_net_weight_path}")
#         else:
#             print("No u_net weight path is provided, using random weights")

#     def get_batch(self, index):
#         # Retrieve batch from dataloader
#         data_iter = iter(self.dataloader)
#         desired_index = index
        
#         try:
#             for i in range(desired_index + 1):
#                 batch = next(data_iter)
#         except StopIteration:
#             print("Index exceeds the number of batches in the DataLoader.")
#             return None
            
#         # Process the data for prediction
#         inputs = batch[0]
#         return inputs.to(self.config.device)

#     def eval(self):
#         # Get all inputs from dataloader
#         inputs = None
#         for batch in self.dataloader:
#             inputs = batch[0].to(self.config.device)
#             break  # Only process the first batch
            
#         if inputs is None:
#             print("No data available for evaluation.")
#             return
            
#         print(f"Input shape: {inputs.shape}")
        
#         complete_prediction = []
#         complete_original = []
        
#         # Process each input in the batch
#         for idx, input_signal in tqdm(enumerate(inputs), desc="Processing"):
#             input_signal = input_signal.unsqueeze(0)  # Add batch dimension
            
#             # Initialize with random noise for parts to be predicted
#             signal = randn_tensor(input_signal.shape, device=self.config.device, dtype=self.unet.dtype)
#             signal[:, :, :self.config.prediction_point] = input_signal[:, :, :self.config.prediction_point]
            
#             # Run diffusion model for prediction
#             predicted_signal = self.pipeline.do_prediction(
#                 signal,
#                 self.config.prediction_point,
#                 batch_size=1,
#                 num_inference_steps=self.config.num_train_timesteps
#             )
            
#             # Clamp values to 0-1 range
#             predicted_signal = predicted_signal.clamp(0, 1)
            
#             # Convert to numpy for processing
#             predicted_numpy = predicted_signal.cpu().numpy()
#             original_numpy = input_signal.cpu().numpy()
            
#             # Denormalize the signals back to original range
#             predicted_denorm = self.dataset.denormalize_with_min_max(predicted_numpy)
#             original_denorm = self.dataset.denormalize_with_min_max(original_numpy)
            
#             # Store results for visualization
#             if idx == 0:
#                 complete_prediction = predicted_denorm[0, 0, :]
#                 complete_original = original_denorm[0, 0, :]
#             else:
#                 # Append only the predicted part to avoid overlap
#                 complete_prediction = np.concatenate((
#                     complete_prediction, 
#                     predicted_denorm[0, 0, self.config.prediction_point:]
#                 ))
#                 complete_original = np.concatenate((
#                     complete_original, 
#                     original_denorm[0, 0, self.config.prediction_point:]
#                 ))
        
#         # Save results
#         cache_dir = "caches/prediction"
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
            
#         np.save(os.path.join(cache_dir, 'pred.npy'), complete_prediction)
#         np.save(os.path.join(cache_dir, 'orig.npy'), complete_original)
        
#         # Visualize results
#         self.visualize_results(complete_original, complete_prediction)
        
#     def visualize_results(self, original, predicted):
#         """
#         Visualize the original and predicted signals.
        
#         Args:
#             original: Original signal data
#             predicted: Predicted signal data
#         """
#         # Create directory for visualizations
#         cache_dir = "caches/prediction"
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
            
#         # Calculate metrics
#         mse = np.mean((predicted - original) ** 2)
#         rmse = np.sqrt(mse)
#         mae = np.mean(np.abs(predicted - original))
        
#         # Calculate R² (coefficient of determination)
#         # R² = 1 - (sum of squared residuals) / (total sum of squares)
#         ss_res = np.sum((original - predicted) ** 2)
#         ss_tot = np.sum((original - np.mean(original)) ** 2)
#         r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
#         print(f"Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")
        
#         # Create plot
#         plt.figure(figsize=(15, 6))
        
#         # Plot both signals
#         x = np.arange(len(original))
#         plt.plot(x, original, label='Original Signal')
#         plt.plot(x, predicted, label='Predicted Signal')
        
#         # Add vertical line at prediction point
#         plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
        
#         # Add title and labels
#         plt.title(f'Original vs Predicted EEG Signal\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}')
#         plt.xlabel('Time Point')
#         plt.ylabel('Amplitude')
#         plt.legend()
        
#         # Save the figure
#         plt.savefig(os.path.join(cache_dir, 'eeg_comparison.png'))
#         plt.close()
        
#         # Also create a plot of error over time
#         error = np.abs(predicted - original)
        
#         plt.figure(figsize=(15, 6))
#         plt.plot(x, error)
#         plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
#         plt.title('Absolute Error Over Time')
#         plt.xlabel('Time Point')
#         plt.ylabel('Absolute Error')
#         plt.legend()
        
#         plt.savefig(os.path.join(cache_dir, 'error_over_time.png'))
#         plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from mmengine import Config
from ..registry import EEGDiffMR, EEGDiffDR
from ..pipeline import DDIMPipeline1D
from ..dataset.EEG_dataset_1d import EvaluationDataset1D

@EEGDiffMR.register_module()
class EEGDiffEvaler1D:
    def __init__(self,
                 unet: Config,
                 dataset: Config,
                 evaler_config: Config,
                 noise_scheduler: Config):
        self.config = evaler_config
        self.unet = EEGDiffMR.build(unet)
        self.initial_unet()
        self.noise_scheduler = EEGDiffMR.build(noise_scheduler)
        
        # Define a custom normalization function
        def normalize_1d(tensor):
            return (tensor - 0.5) / 0.5  # Equivalent to Normalize(mean=[0.5], std=[0.5])
        
        # Create dataset for evaluation
        self.dataset = EvaluationDataset1D(
            csv_path=self.config.csv_path,
            window_size=self.config.window_size,
            prediction_point=self.config.prediction_point
        )
        self.dataset.transform = normalize_1d
        
        # Setup pipeline for inference
        self.pipeline = DDIMPipeline1D(unet=self.unet, scheduler=self.noise_scheduler)
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset=self.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )

    def initial_unet(self):
        self.unet.to(self.config.device)
        if self.config.u_net_weight_path is not None:
            self.unet.load_state_dict(torch.load(self.config.u_net_weight_path, map_location=self.config.device))
            print(f"Load u_net weight from {self.config.u_net_weight_path}")
        else:
            print("No u_net weight path is provided, using random weights")

    def eval(self):
        # Get all inputs from dataloader
        inputs = None
        for batch in self.dataloader:
            inputs = batch[0].to(self.config.device)
            break  # Only process the first batch for simplicity
            
        if inputs is None:
            print("No data available for evaluation.")
            return
            
        print(f"Input shape: {inputs.shape}")
        
        complete_prediction = []
        complete_original = []
        
        # Process each input in the batch
        for idx, input_signal in tqdm(enumerate(inputs), desc="Processing signals"):
            input_signal = input_signal.unsqueeze(0)  # Add batch dimension
            
            # Initialize with random noise for parts to be predicted
            signal = randn_tensor(input_signal.shape, device=self.config.device, dtype=self.unet.dtype)
            signal[:, :, :self.config.prediction_point] = input_signal[:, :, :self.config.prediction_point]
            
            # Run diffusion model for prediction
            predicted_signal = self.pipeline.do_prediction(
                signal,
                self.config.prediction_point,
                batch_size=1,
                num_inference_steps=self.config.num_train_timesteps
            )
            
            # Clamp values to 0-1 range
            predicted_signal = predicted_signal.clamp(0, 1)
            
            # Move to CPU before converting to numpy
            predicted_numpy = predicted_signal.cpu().numpy()
            original_numpy = input_signal.cpu().numpy()
            
            # Denormalize the signals back to original range
            predicted_denorm = self.dataset.denormalize_with_min_max(predicted_numpy)
            original_denorm = self.dataset.denormalize_with_min_max(original_numpy)
            
            # Store results for visualization
            if idx == 0:
                complete_prediction = predicted_denorm[0, 0, :]
                complete_original = original_denorm[0, 0, :]
            else:
                # Append only the predicted part to avoid overlap
                complete_prediction = np.concatenate((
                    complete_prediction, 
                    predicted_denorm[0, 0, self.config.prediction_point:]
                ))
                complete_original = np.concatenate((
                    complete_original, 
                    original_denorm[0, 0, self.config.prediction_point:]
                ))
        
        # Save results
        cache_dir = "caches/prediction"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        np.save(os.path.join(cache_dir, 'pred.npy'), complete_prediction)
        np.save(os.path.join(cache_dir, 'orig.npy'), complete_original)
        
        # Visualize results
        self.visualize_results(complete_original, complete_prediction)
        
    def visualize_results(self, original, predicted):
        """
        Visualize the original and predicted signals.
        
        Args:
            original: Original signal data
            predicted: Predicted signal data
        """
        # Create directory for visualizations
        cache_dir = "caches/prediction"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Calculate metrics
        mse = np.mean((predicted - original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted - original))
        
        # Calculate R² (coefficient of determination)
        # R² = 1 - (sum of squared residuals) / (total sum of squares)
        ss_res = np.sum((original - predicted) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print(f"Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")
        
        # Create plot
        plt.figure(figsize=(15, 6))
        
        # Plot both signals
        x = np.arange(len(original))
        plt.plot(x, original, label='Original Signal')
        plt.plot(x, predicted, label='Predicted Signal')
        
        # Add vertical line at prediction point
        plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
        
        # Add title and labels
        plt.title(f'Original vs Predicted EEG Signal\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}')
        plt.xlabel('Time Point')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Save the figure
        plt.savefig(os.path.join(cache_dir, 'eeg_comparison.png'))
        plt.close()
        
        # Also create a plot of error over time
        error = np.abs(predicted - original)
        
        plt.figure(figsize=(15, 6))
        plt.plot(x, error)
        plt.axvline(x=self.config.prediction_point, color='r', linestyle='--', label='Prediction Point')
        plt.title('Absolute Error Over Time')
        plt.xlabel('Time Point')
        plt.ylabel('Absolute Error')
        plt.legend()
        
        plt.savefig(os.path.join(cache_dir, 'error_over_time.png'))
        plt.close()