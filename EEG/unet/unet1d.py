# from dataclasses import dataclass
# from typing import Optional, Tuple, Union

# import torch
# import torch.nn as nn

# from diffusers.configuration_utils import ConfigMixin, register_to_config
# from diffusers.utils import BaseOutput
# from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
# from diffusers.models.modeling_utils import ModelMixin
# from ..registry import EEGDiffMR

# # Custom 1D blocks
# class DownBlock1D(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         temb_channels,
#         dropout=0.0,
#         num_layers=1,
#         resnet_eps=1e-5,
#         resnet_act_fn="silu",
#         resnet_groups=32,
#         add_downsample=True,
#         downsample_padding=1,
#     ):
#         super().__init__()
#         self.resnet_blocks = nn.ModuleList([
#             ResnetBlock1D(
#                 in_channels=in_channels if i == 0 else out_channels,
#                 out_channels=out_channels,
#                 temb_channels=temb_channels,
#                 dropout=dropout,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 act_fn=resnet_act_fn,
#             )
#             for i in range(num_layers)
#         ])

#         self.downsamplers = None
#         if add_downsample:
#             self.downsamplers = nn.ModuleList([
#                 Downsample1D(
#                     out_channels, out_channels=out_channels, padding=downsample_padding
#                 )
#             ])

#     def forward(self, hidden_states, temb=None):
#         output_states = ()

#         for resnet_block in self.resnet_blocks:
#             hidden_states = resnet_block(hidden_states, temb)
#             output_states += (hidden_states,)

#         if self.downsamplers is not None:
#             for downsampler in self.downsamplers:
#                 hidden_states = downsampler(hidden_states)

#             output_states += (hidden_states,)

#         return hidden_states, output_states


# class UpBlock1D(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         prev_output_channel,
#         out_channels,
#         temb_channels,
#         dropout=0.0,
#         num_layers=1,
#         resnet_eps=1e-5,
#         resnet_act_fn="silu",
#         resnet_groups=32,
#         add_upsample=True,
#     ):
#         super().__init__()
#         self.resnet_blocks = nn.ModuleList([
#             ResnetBlock1D(
#                 in_channels=in_channels if i == 0 else out_channels,
#                 out_channels=out_channels,
#                 temb_channels=temb_channels,
#                 dropout=dropout,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 act_fn=resnet_act_fn,
#             )
#             for i in range(num_layers)
#         ])

#         self.upsamplers = None
#         if add_upsample:
#             self.upsamplers = nn.ModuleList([Upsample1D(out_channels, out_channels=out_channels)])

#     def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
#         for resnet_block in self.resnet_blocks:
#             # Pop res hidden states
#             res_hidden_states = res_hidden_states_tuple[-1]
#             res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
#             # Check channel dimensions and adjust if needed
#             if hidden_states.shape[1] + res_hidden_states.shape[1] != resnet_block.in_channels:
#                 # Print debug info
#                 print(f"Hidden states shape: {hidden_states.shape}")
#                 print(f"Res hidden states shape: {res_hidden_states.shape}")
#                 print(f"Expected input channels for resnet_block: {resnet_block.in_channels}")
                
#                 # Adjust channels - either through projection or padding
#                 if hasattr(resnet_block, 'channel_projection'):
#                     # Use a projection if available
#                     hidden_states = resnet_block.channel_projection(hidden_states)
#                 else:
#                     # Otherwise, adjust channels through a 1x1 convolution
#                     adjust_conv = torch.nn.Conv1d(
#                         hidden_states.shape[1], 
#                         resnet_block.in_channels - res_hidden_states.shape[1],
#                         kernel_size=1
#                     ).to(hidden_states.device)
#                     hidden_states = adjust_conv(hidden_states)
            
#             # Concatenate along channel dimension
#             hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
#             hidden_states = resnet_block(hidden_states, temb)

#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states)

#         return hidden_states


# class UNetMidBlock1D(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         temb_channels,
#         dropout=0.0,
#         resnet_eps=1e-5,
#         resnet_act_fn="silu",
#         output_scale_factor=1.0,
#         resnet_time_scale_shift="default",
#         resnet_groups=32,
#         add_attention=True,
#     ):
#         super().__init__()
#         self.resnet_blocks = nn.ModuleList([
#             ResnetBlock1D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 dropout=dropout,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 act_fn=resnet_act_fn,
#             ),
#             ResnetBlock1D(
#                 in_channels=in_channels,
#                 out_channels=in_channels,
#                 temb_channels=temb_channels,
#                 dropout=dropout,
#                 eps=resnet_eps,
#                 groups=resnet_groups,
#                 act_fn=resnet_act_fn,
#             ),
#         ])

#     def forward(self, hidden_states, temb=None):
#         for resnet_block in self.resnet_blocks:
#             hidden_states = resnet_block(hidden_states, temb)

#         return hidden_states


# class ResnetBlock1D(nn.Module):
#     # def __init__(
#     #     self,
#     #     in_channels,
#     #     out_channels=None,
#     #     temb_channels=512,
#     #     dropout=0.0,
#     #     groups=32,
#     #     eps=1e-5,
#     #     act_fn="silu",
#     # ):
#     #     super().__init__()
#     #     self.in_channels = in_channels
#     #     self.out_channels = in_channels if out_channels is None else out_channels

#     #     self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
#     #     self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#     #     if temb_channels is not None:
#     #         self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
#     #     else:
#     #         self.time_emb_proj = None

#     #     self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
#     #     self.dropout = torch.nn.Dropout(dropout)
#     #     self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

#     #     if self.in_channels != self.out_channels:
#     #         self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

#     #     self.act_fn = get_activation(act_fn)

  
#     def __init__(
#         self,
#         in_channels,
#         out_channels=None,
#         temb_channels=512,
#         dropout=0.0,
#         groups=32,
#         eps=1e-5,
#         act_fn="silu",
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = in_channels if out_channels is None else out_channels

#         # Ensure number of groups divides the number of channels
#         # Groups should divide channels evenly
#         while self.in_channels % groups != 0 and groups > 1:
#             groups = groups // 2
        
#         self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
#         self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

#         if temb_channels is not None:
#             self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
#         else:
#             self.time_emb_proj = None

#         # Ensure number of groups divides the number of channels
#         while self.out_channels % groups != 0 and groups > 1:
#             groups = groups // 2
            
#         self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

#         if self.in_channels != self.out_channels:
#             self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

#         if act_fn == "silu":
#             self.act_fn = nn.SiLU()
#         elif act_fn == "relu":
#             self.act_fn = nn.ReLU()
#         else:
#             self.act_fn = nn.SiLU()  # Default to SiLU

#     def forward(self, input_tensor, temb=None):
#         hidden_states = input_tensor
#         hidden_states = self.norm1(hidden_states)
#         hidden_states = self.act_fn(hidden_states)
#         hidden_states = self.conv1(hidden_states)

#         if temb is not None and self.time_emb_proj is not None:
#             temb = self.act_fn(temb)
#             temb = self.time_emb_proj(temb)[:, :, None]
#             hidden_states = hidden_states + temb

#         hidden_states = self.norm2(hidden_states)
#         hidden_states = self.act_fn(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.conv2(hidden_states)

#         if self.in_channels != self.out_channels:
#             input_tensor = self.conv_shortcut(input_tensor)

#         output_tensor = input_tensor + hidden_states
#         return output_tensor


# class Upsample1D(nn.Module):
#     def __init__(self, channels, out_channels=None, use_conv=True):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.use_conv = use_conv

#         self.conv = None
#         if use_conv:
#             self.conv = torch.nn.Conv1d(self.channels, self.out_channels, kernel_size=3, padding=1)

#     def forward(self, hidden_states):
#         # Upsample by factor of 2
#         hidden_states = nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="linear")

#         if self.use_conv:
#             hidden_states = self.conv(hidden_states)

#         return hidden_states


# class Downsample1D(nn.Module):
#     def __init__(self, channels, out_channels=None, padding=1):
#         super().__init__()
#         self.channels = channels
#         self.out_channels = out_channels or channels
#         self.padding = padding

#         self.conv = torch.nn.Conv1d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding)

#     def forward(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         return hidden_states


# def get_activation(act_fn):
#     if act_fn == "silu":
#         return nn.SiLU()
#     elif act_fn == "relu":
#         return nn.ReLU()
#     elif act_fn == "leaky_relu":
#         return nn.LeakyReLU()
#     else:
#         raise ValueError(f"Unsupported activation: {act_fn}")


# @dataclass
# class UNet1DOutput(BaseOutput):
#     """
#     The output of [`UNet1DModel`].

#     Args:
#         sample (`torch.FloatTensor` of shape `(batch_size, num_channels, length)`):
#             The hidden states output from the last layer of the model.
#     """
#     sample: torch.FloatTensor


# @EEGDiffMR.register_module()
# class UNet1DModel(ModelMixin, ConfigMixin):
#     """
#     A 1D UNet model for processing EEG signals.
#     """
#     @register_to_config
#     def __init__(
#         self,
#         sample_size: Optional[int] = None,
#         in_channels: int = 1,
#         out_channels: int = 1,
#         center_input_sample: bool = False,
#         time_embedding_type: str = "positional",
#         freq_shift: int = 0,
#         flip_sin_to_cos: bool = True,
#         down_block_types: Tuple[str] = ("DownBlock1D", "DownBlock1D", "DownBlock1D", "DownBlock1D"),
#         up_block_types: Tuple[str] = ("UpBlock1D", "UpBlock1D", "UpBlock1D", "UpBlock1D"),
#         block_out_channels: Tuple[int] = (64, 128, 256, 512),
#         layers_per_block: int = 2,
#         mid_block_scale_factor: float = 1,
#         downsample_padding: int = 1,
#         act_fn: str = "silu",
#         norm_num_groups: int = 32,
#         norm_eps: float = 1e-5,
#         add_attention: bool = False,
#         dropout: float = 0.0,
#     ):
#         super().__init__()

#         self.sample_size = sample_size
#         time_embed_dim = block_out_channels[0] * 4

#         # Check inputs
#         if len(down_block_types) != len(up_block_types):
#             raise ValueError(
#                 f"Must provide the same number of `down_block_types` as `up_block_types`. "
#                 f"`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
#             )

#         if len(block_out_channels) != len(down_block_types):
#             raise ValueError(
#                 f"Must provide the same number of `block_out_channels` as `down_block_types`. "
#                 f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
#             )

#         # input
#         self.conv_in = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

#         # time
#         if time_embedding_type == "fourier":
#             self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
#             timestep_input_dim = 2 * block_out_channels[0]
#         elif time_embedding_type == "positional":
#             self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
#             timestep_input_dim = block_out_channels[0]

#         self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

#         self.down_blocks = nn.ModuleList([])
#         self.mid_block = None
#         self.up_blocks = nn.ModuleList([])

#         # down
#         output_channel = block_out_channels[0]
#         for i, down_block_type in enumerate(down_block_types):
#             input_channel = output_channel
#             output_channel = block_out_channels[i]
#             is_final_block = i == len(block_out_channels) - 1

#             down_block = DownBlock1D(
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 temb_channels=time_embed_dim,
#                 dropout=dropout,
#                 num_layers=layers_per_block,
#                 resnet_eps=norm_eps,
#                 resnet_groups=norm_num_groups,
#                 resnet_act_fn=act_fn,
#                 add_downsample=not is_final_block,
#                 downsample_padding=downsample_padding,
#             )
#             self.down_blocks.append(down_block)

#         # mid
#         self.mid_block = UNetMidBlock1D(
#             in_channels=block_out_channels[-1],
#             temb_channels=time_embed_dim,
#             dropout=dropout,
#             resnet_eps=norm_eps,
#             resnet_act_fn=act_fn,
#             output_scale_factor=mid_block_scale_factor,
#             resnet_groups=norm_num_groups,
#             add_attention=add_attention,
#         )

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, up_block_type in enumerate(up_block_types):
#             prev_output_channel = output_channel
#             output_channel = reversed_block_out_channels[i]
#             input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

#             is_final_block = i == len(block_out_channels) - 1

#             up_block = UpBlock1D(
#                 in_channels=input_channel * 2,  # multiply by 2 because of skip connections
#                 prev_output_channel=prev_output_channel,
#                 out_channels=output_channel,
#                 temb_channels=time_embed_dim,
#                 dropout=dropout,
#                 num_layers=layers_per_block + 1,
#                 resnet_eps=norm_eps,
#                 resnet_act_fn=act_fn,
#                 resnet_groups=norm_num_groups,
#                 add_upsample=not is_final_block,
#             )
#             self.up_blocks.append(up_block)
#             prev_output_channel = output_channel

#         # out
#         num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv1d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

#     def forward(
#         self,
#         sample: torch.FloatTensor,
#         timestep: Union[torch.Tensor, float, int],
#         return_dict: bool = True,
#     ) -> Union[UNet1DOutput, Tuple]:
#         # 0. center input if necessary
#         if self.config.center_input_sample:
#             sample = 2 * sample - 1.0

#         # 1. time
#         timesteps = timestep
#         if not torch.is_tensor(timesteps):
#             timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
#         elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
#             timesteps = timesteps[None].to(sample.device)

#         # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#         timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

#         t_emb = self.time_proj(timesteps)
#         t_emb = t_emb.to(dtype=self.dtype)
#         emb = self.time_embedding(t_emb)

#         # 2. pre-process
#         sample = self.conv_in(sample)

#         # 3. down
#         down_block_res_samples = (sample,)
#         for downsample_block in self.down_blocks:
#             sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
#             down_block_res_samples += res_samples

#         # 4. mid
#         sample = self.mid_block(sample, emb)

#         # 5. up
#         for upsample_block in self.up_blocks:
#             res_samples = down_block_res_samples[-len(upsample_block.resnet_blocks):]
#             down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnet_blocks)]
#             sample = upsample_block(sample, res_samples, emb)

#         # 6. post-process
#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         if not return_dict:
#             return (sample,)

#         return UNet1DOutput(sample=sample)


from diffusers import UNet1DModel
from ..registry import EEGDiffMR

@EEGDiffMR.register_module()
class UNet1DModelWrapper(UNet1DModel):
    """
    A wrapper around diffusers.UNet1DModel that registers with the EEG-DIF registry.
    This allows us to use the standard diffusers implementation while maintaining
    compatibility with the EEG-DIF architecture.
    
    All parameters and functionality are inherited from the parent UNet1DModel class.
    """
    pass