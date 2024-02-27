import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image, ImageOps
import numpy as np

class ImageDirIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {}),
                "image_index": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_image_by_index"
    CATEGORY = "cspnodes"

    def get_image_by_index(self, directory_path, image_index):
        # Get list of image files sorted by modification time (most recent first)
        image_files = sorted(
            [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))],
            key=lambda x: os.path.getmtime(x),
            reverse=True
        )

        # Validate index
        if image_index < 0 or image_index >= len(image_files):
            raise IndexError("Image index out of range.")

        # Load and preprocess the image
        image = Image.open(image_files[image_index])
        image = ImageOps.exif_transpose(image)  # Correct orientation
        image = image.convert("RGB")  # Ensure image is in RGB format

        # Convert image to tensor
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

        return (image_tensor,)

class Modelscopet2v:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {}),
                "negative_prompt": ("STRING", {"default": None}),
                "model_path": ("STRING", {"default": "cerspense/zeroscope_v2_576w"}),
                "num_inference_steps": ("INT", {"default": 25}),
                "guidance_scale": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 42}),
                "width": ("INT", {"default": 576}),
                "height": ("INT", {"default": 320}),
                "num_frames": ("INT", {"default": 24}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_video_frames"
    CATEGORY = "cspnodes/modelscope"

    def generate_video_frames(self, prompt, model_path, num_inference_steps, height, width, num_frames, guidance_scale, negative_prompt, seed):
        # Set up the generator for deterministic results if seed is provided
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Added generator to the pipe call
        video_frames = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width, num_frames=num_frames, guidance_scale=guidance_scale, negative_prompt=negative_prompt, generator=generator).frames
        
        # Ensure video_frames is a PyTorch tensor
        if not isinstance(video_frames, torch.Tensor):
            video_frames = torch.tensor(video_frames, dtype=torch.float32)

        # Remove the unnecessary batch dimension explicitly and permute the dimensions
        # The expected shape is (num_frames, height, width, channels)
        video_frames = video_frames.squeeze(0).permute(0, 1, 2, 3)

        # Convert the tensor to CPU and to uint8 if it's not already
        video_frames = video_frames.to('cpu')

        # return (video_frames_numpy,)
        return (video_frames,)

class Modelscopev2v:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "negative_prompt": ("STRING", {"default": None}),
                "model_path": ("STRING", {"default": "cerspense/zeroscope_v2_XL"}),  
                "strength": ("FLOAT", {"default": 0.70}),
                "num_inference_steps": ("INT", {"default": 25}),
                "guidance_scale": ("FLOAT", {"default": 8.50}),
                "seed": ("INT", {"default": 42}),
                "enable_forward_chunking": ("BOOLEAN", {"default": False}),
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_video_frames"
    CATEGORY = "cspnodes/modelscope"

    def transform_video_frames(self, video_frames, prompt, model_path, strength, num_inference_steps, guidance_scale, negative_prompt, seed, enable_forward_chunking, enable_vae_slicing):
        # Set up the generator for deterministic results if seed is provided
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        # Initialize the diffusion pipeline with the specified model path
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Apply memory optimizations based on the toggles
        if enable_forward_chunking:
            pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        if enable_vae_slicing:
            pipe.enable_vae_slicing()

        # Convert tensor to list of PIL Images
        # Assuming video_frames is a float tensor with values in [0, 1]
        video_frames_uint8 = (video_frames * 255).byte()
        video = [Image.fromarray(frame.numpy(), 'RGB') for frame in video_frames_uint8]

        # Generate new video frames
        video_frames = pipe(prompt, video=video, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, generator=generator).frames

        # Ensure video_frames is a PyTorch tensor
        if not isinstance(video_frames, torch.Tensor):
            video_frames = torch.tensor(video_frames, dtype=torch.float32)

        # The expected shape is (num_frames, height, width, channels)
        video_frames = video_frames.squeeze(0).permute(0, 1, 2, 3)

        # Convert the tensor to CPU and to uint8 if it's not already
        video_frames = video_frames.to('cpu')

        # return (video_frames_numpy,)
        return (video_frames,)
    
NODE_CLASS_MAPPINGS = {
    "ImageDirIterator": ImageDirIterator,
    "Modelscopet2v": Modelscopet2v,
    "Modelscopev2v": Modelscopev2v,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDirIterator": "Image Dir Iterator",
    "Modelscopet2v": "Modelscope t2v",
    "Modelscopev2v": "Modelscope v2v",
}