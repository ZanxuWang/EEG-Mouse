from mmengine import read_base
from EEG import EEGDiffTrainner1D

with read_base():
    from ._base_.train_dataset_1d import train_dataset, val_dataset
    from ._base_.unet_1d import unet
    from ._base_.noise_scheduler import noise_scheduler
    from ._base_.basic_information import prject_name

device = "cuda"  # Change to "cpu" if no GPU available

train_config = dict(
    device=device,
    output_dir="C:/Users/chris/EEG-DIF/outputs/1d_model",
    u_net_weight_path=None,  # Set to path if you have pretrained weights
    prediction_point=934,   # Start prediction from 50% of signal
    num_train_timesteps=200,
    num_epochs=20,
    train_batch_size=16,     # Adjust based on your GPU memory
    eval_batch_size=16,
    learning_rate=0.00002,
    lr_warmup_steps=5,
    eval_begin=100,          # Start evaluation after this many iterations
    eval_interval=200,       # Evaluate every 100 iterations
)

project_name = "EEG-DIF-1D"

trainner = dict(
    type=EEGDiffTrainner1D,
    trainner_config=train_config,
    unet=unet,
    noise_scheduler=noise_scheduler,
    optimizer=dict(type="Adam", learning_rate=0.00002),  # Same as in train_config
    train_dataset=train_dataset,
    val_dataset=val_dataset)

wandb_config = dict(
    learning_rate=trainner['optimizer']['learning_rate'],
    architecture="1D diffusion model",
    dataset=project_name,
    epochs=train_config['num_epochs'],
)