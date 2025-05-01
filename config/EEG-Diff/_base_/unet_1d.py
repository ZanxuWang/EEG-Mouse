# from EEG import UNet1DModelWrapper

# unet = dict(
#     type=UNet1DModelWrapper,
#     down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
#     up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
#     block_out_channels=(64, 128, 256, 512),
#     in_channels=1,
#     out_channels=1,
#     sample_size=2560,  # Length of your EEG signals
#     layers_per_block=2,
#     norm_num_groups=8,  # Reduced from 32 as 1D signals may have fewer channels
#     add_attention=False,  # Set to False for 1D signals
#     dropout=0.2,
# )


from EEG import UNet1DModelWrapper  # Changed from UNet1DModel to UNet1DModelWrapper

unet = dict(
    type=UNet1DModelWrapper,
    sample_size=1280,  # Length of your EEG signals
    in_channels=1,     # Single channel EEG data
    out_channels=1,    # Output also single channel
    layers_per_block=2,
    block_out_channels=(32, 64, 128, 256),
    norm_num_groups=4,  # Reduced for 1D signals
    down_block_types=("DownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D", "AttnDownBlock1D"),
    up_block_types=("AttnUpBlock1D", "AttnUpBlock1D", "AttnUpBlock1D", "UpBlock1D"),

    #mid_block_type="MidBlock1D",
    # Additional parameters supported by diffusers.UNet1DModel
    time_embedding_type="positional",
    act_fn="silu",
    #attention_head_dim=None,  # No attention for 1D EEG signals
    #norm_eps=1e-5,
    #resnet_time_scale_shift="default",
)