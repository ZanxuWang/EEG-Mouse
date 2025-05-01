import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from ..registry import EEGDiffDR

@EEGDiffDR.register_module()
class EEGDataset1D(Dataset):
    def __init__(self, csv_path, sequence_length=2560):
        """
        Args:
            csv_path (str): Path to the CSV file containing the dataset.
            sequence_length (int): The length of each sequence to be processed.
            step_size (int): Step size for sliding window, if using overlapping segments.
        """
        self.transform = None
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        #self.step_size = step_size
        
        # Load data from CSV - assuming no headers and each row is a sample
        data = pd.read_csv(csv_path, header=None)
        self.data = data.values
        
        # Calculate normalization values
        self.normalized_data, \
        self.max_value, \
        self.min_value = self.normalize_with_min_max(self.data)
        
        # Calculate effective number of segments
        self.num_samples = self.data.shape[0]  # Number of samples/rows in CSV

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Get the entire sample as a sequence
        sequence = self.normalized_data[index, :]
        
        # Convert to tensor
        sequence = torch.from_numpy(sequence).float()
        
        # Reshape to [channels, sequence_length]
        sequence = sequence.reshape(1, -1)  # 1 channel
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return (sequence,)
    
    def normalize_with_min_max(self, data):
        max_values = np.max(data)
        min_values = np.min(data)
        
        normalized_data = np.zeros_like(data, dtype=float)
        if max_values == min_values:
            normalized_data[:] = 0.0
        else:
            normalized_data = (data - min_values) / (max_values - min_values)
            
        return normalized_data, max_values, min_values
    
    def denormalize_with_min_max(self, normalized_data):
        denormalized_data = normalized_data * (self.max_value - self.min_value) + self.min_value
        return denormalized_data


@EEGDiffDR.register_module()
class PredictionEEGDataset1D(Dataset):
    def __init__(self, csv_path, sequence_length=2560, prediction_length=1280):
        """
        Dataset for long-term prediction tasks.
        
        Args:
            csv_path (str): Path to the CSV file containing the dataset.
            sequence_length (int): Total length of sequence.
            prediction_length (int): Length of the prediction window.
            step_size (int): Step size for sliding window.
        """
        self.transform = None
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        #self.step_size = step_size
        
        # Load data from CSV - assuming no headers and each row is a sample
        data = pd.read_csv(csv_path, header=None)
        self.data = data.values
        
        # Calculate normalization values
        self.normalized_data, \
        self.max_value, \
        self.min_value = self.normalize_with_min_max(self.data)
        
        # Calculate effective number of segments
        self.num_samples = self.data.shape[0]  # Number of samples/rows in CSV

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        # Get the entire sample as a sequence
        sequence = self.normalized_data[index, :]
        
        # Convert to tensor
        sequence = torch.from_numpy(sequence).float()
        
        # Reshape to [channels, sequence_length]
        sequence = sequence.reshape(1, -1)  # 1 channel
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return (sequence,)
    
    def normalize_with_min_max(self, data):
        max_values = np.max(data)
        min_values = np.min(data)
        
        normalized_data = np.zeros_like(data, dtype=float)
        if max_values == min_values:
            normalized_data[:] = 0.0
        else:
            normalized_data = (data - min_values) / (max_values - min_values)
            
        return normalized_data, max_values, min_values
    
    def denormalize_with_min_max(self, normalized_data):
        denormalized_data = normalized_data * (self.max_value - self.min_value) + self.min_value
        return denormalized_data


@EEGDiffDR.register_module()
class EvaluationDataset1D(Dataset):
    def __init__(self, csv_path, window_size=2560, prediction_point=1280):
        """
        Dataset for evaluation with sliding window approach.
        
        Args:
            csv_path (str): Path to the CSV file containing the dataset.
            window_size (int): Size of the window for each data item.
            prediction_point (int): Number of points to predict.
            step_size (int): Step size to move the window.
        """
        self.transform = None
        self.csv_path = csv_path
        self.window_size = window_size
        self.prediction_point = prediction_point
        #self.step_size = step_size
        
        # Load data from CSV - assuming no headers and each row is a sample
        data = pd.read_csv(csv_path, header=None)
        self.data = data.values
        
        # Calculate normalization values
        self.normalized_data, \
        self.max_value, \
        self.min_value = self.normalize_with_min_max(self.data)
        
        # Calculate effective number of segments per sample
        self.segments_per_sample = (self.data.shape[1] - self.window_size) // self.step_size + 1
        
        # Total number of segments across all samples
        self.total_segments = self.data.shape[0] * self.segments_per_sample

    def __len__(self):
        return self.total_segments
    
    def __getitem__(self, index):
        # Calculate which sample and which segment within that sample
        sample_idx = index // self.segments_per_sample
        segment_idx = index % self.segments_per_sample
        
        # Calculate start position
        start_pos = segment_idx * self.step_size
        
        # Extract the window from the sample
        window = self.normalized_data[sample_idx, start_pos:start_pos + self.window_size]
        
        # Convert to tensor
        window = torch.from_numpy(window).float()
        
        # Reshape to [channels, sequence_length]
        window = window.reshape(1, -1)  # 1 channel
        
        if self.transform:
            window = self.transform(window)
            
        return (window,)
    
    def normalize_with_min_max(self, data):
        max_values = np.max(data)
        min_values = np.min(data)
        
        normalized_data = np.zeros_like(data, dtype=float)
        if max_values == min_values:
            normalized_data[:] = 0.0
        else:
            normalized_data = (data - min_values) / (max_values - min_values)
            
        return normalized_data, max_values, min_values
    
    def denormalize_with_min_max(self, normalized_data):
        denormalized_data = normalized_data * (self.max_value - self.min_value) + self.min_value
        return denormalized_data