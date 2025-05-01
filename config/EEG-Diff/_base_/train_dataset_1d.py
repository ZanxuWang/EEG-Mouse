
from EEG import EEGDataset1D, PredictionEEGDataset1D

train_dataset = dict(
    type=EEGDataset1D,
    csv_path="C:/Users/chris/EEG-DIF/data/train.csv", 
    sequence_length=1280,  # Full length of your EEG signals
    #step_size=128,         # Optional step size for overlapping segments
)

val_dataset = dict(
    type=EEGDataset1D,
    csv_path="C:/Users/chris/EEG-DIF/data/test.csv",  # Using a portion of train data for validation
    sequence_length=1280,
    #step_size=128,
)