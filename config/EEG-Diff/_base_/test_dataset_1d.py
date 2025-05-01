
from EEG import EEGDataset1D, PredictionEEGDataset1D, EvaluationDataset1D

test_dataset = dict(
    type=EvaluationDataset1D,
    csv_path="C:/Users/chris/EEG-DIF/data/test.csv",
    window_size=1280,          # Full signal length
    prediction_point=934,     # Start predicting from this point (50% of the signal)
    #step_size=1280,            # Step size for sliding window
)