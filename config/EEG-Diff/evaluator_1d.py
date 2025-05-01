from mmengine import read_base
from EEG import EEGDiffEvaler1D

with read_base():
    from ._base_.test_dataset_1d import test_dataset
    from ._base_.unet_1d import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"  # Change to "cpu" if no GPU available

dataset = dict(
    type=test_dataset['type'],
    csv_path=test_dataset['csv_path'],
)

eval_config = dict(
    device=device,
    csv_path="C:/Users/chris/EEG-DIF/data/test.csv",  # Test data path
    output_dir="C:/Users/chris/EEG-DIF/outputs/1d_model",
    u_net_weight_path="C:/Users/chris/EEG-DIF/outputs/1d_model/10000/unet.pth",  # Update this path after training
    prediction_point=934,  # Start prediction from 50% of signal
    num_train_timesteps=1000,
    window_size=1280,  # Full signal length
    batch_size=8,
)

project_name = "EEG-DIF-1D"

evaler = dict(
    type=EEGDiffEvaler1D,
    evaler_config=eval_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    dataset=dataset,
)