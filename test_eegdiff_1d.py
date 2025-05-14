from EEG import EEGDiffEvaler1D, EEGDiffMR
from mmengine import Config
import os

# Create necessary directories
os.makedirs("C:/Users/chris/EEG-DIF/caches/prediction", exist_ok=True)

# Load configuration
config_file_path = 'config/EEG-Diff/evaluator_1d.py'
config = Config.fromfile(config_file_path)

# Build evaluator
print("Initializing 1D EEG diffusion evaluator...")
evaler = EEGDiffMR.build(config.evaler)

# Perform evaluation
print("Starting evaluation...")
evaler.eval()
print("Evaluation completed! Results saved to caches/prediction/ directory.")