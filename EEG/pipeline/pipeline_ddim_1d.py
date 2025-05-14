from typing import List, Optional, Tuple, Union
import torch
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline, ImagePipelineOutput

class DDIMPipeline1D(DiffusionPipeline):
    """
    DDIM Pipeline for 1D EEG data.
    
    Parameters:
        unet: U-Net architecture to denoise the encoded signal.
        scheduler: A scheduler to be used in combination with unet to denoise the encoded signal.
    """

    def __init__(self, unet, scheduler):
        super().__init__()

        # Convert scheduler to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            initial_signal,
            prediction_point,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 1000,
            use_clipped_model_output: Optional[bool] = None,
            return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Args:
            initial_signal: The initial signal to condition on.
            prediction_point: Point from which to start prediction.
            batch_size: The number of samples to generate.
            generator: Torch generator(s) for deterministic generation.
            eta: Controls scale of variance (0 is DDIM, 1 is one type of DDPM).
            num_inference_steps: Number of denoising steps.
            use_clipped_model_output: Whether to clip model output.
            return_dict: Whether to return as a dictionary.

        Returns:
            Generated signal.
        """
        # Sample Gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            signal_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size)
        else:
            signal_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Update this part with scaled noise
        noise_scale = 1  # Use same value as in training
        signal = noise_scale * randn_tensor(signal_shape, generator=generator, 
                                        device=self.device, dtype=self.unet.dtype)
        signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. Predict noise model_output
            model_output = self.unet(signal, t).sample

            # 2. Predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
            signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

        # Normalize signal
        signal = (signal / 2 + 0.5).clamp(0, 1)
        
        if not return_dict:
            return (signal,)

        # Return as ImagePipelineOutput for compatibility
        return ImagePipelineOutput(images=signal)

    @torch.no_grad()
    def do_prediction(
            self,
            initial_signal,
            prediction_point: int,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 100,
            use_clipped_model_output: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Performs prediction starting from a specific point.
        
        Args:
            initial_signal: The initial signal to condition on.
            predict_begin_point: Point from which to start prediction.
            batch_size: The number of samples to generate.
            generator: Torch generator(s) for deterministic generation.
            eta: Controls scale of variance (0 is DDIM, 1 is one type of DDPM).
            num_inference_steps: Number of denoising steps.
            use_clipped_model_output: Whether to clip model output.

        Returns:
            Predicted signal.
        """
        # # Clone the initial signal
        # signal = initial_signal.clone()

        # def do_prediction(self, initial_signal, predict_begin_point: int, ...):
        # Instead of just cloning, start with random noise like in __call__


        noise_scale = 1  # Use same value as in training
        signal = noise_scale * randn_tensor(initial_signal.shape, generator=generator, 
                                      device=self.device, dtype=self.unet.dtype)
        signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. Predict noise model_output
            model_output = self.unet(signal, t).sample

            # 2. Predict previous mean of signal x_t-1 and add variance depending on eta
            signal = self.scheduler.step(
                model_output, t, signal, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample
            signal[:, :, :prediction_point] = initial_signal[:, :, :prediction_point]

        return signal