from EEG import EEGDiffTrainner1D, EEGDiffMR
from mmengine import Config
import wandb
import os

# Create necessary directories
os.makedirs("C:/Users/chris/EEG-DIF/outputs/1d_model", exist_ok=True)
os.makedirs("C:/Users/chris/EEG-DIF/data", exist_ok=True)

# Load configuration
config_file_path = 'config/EEG-Diff/trainer_1d.py'
config = Config.fromfile(config_file_path)

# Build trainer
trainner = EEGDiffMR.build(config.trainner)

# Initialize wandb for tracking (optional)
# If you don't want to use wandb, comment out these lines
key = 'bde34833a8dd1b2d2dbdc205886b208888f11039'  # Fill your wandb key here if you want to use it
if key:
    wandb.login(key=key)
    wandb.init(
        project=config.project_name,
        name='EEG-DIF-1D Training',
        config=config.wandb_config
    )
else:
    print("No wandb key provided, training without wandb logging")

# Start training
print("Starting training for 1D EEG diffusion model...")
trainner.train()
print("Training completed!")