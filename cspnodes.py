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
                "num_inference_steps": ("INT", {"default": 40}),
                "height": ("INT", {"default": 320}),
                "width": ("INT", {"default": 576}),
                "num_frames": ("INT", {"default": 24}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_video_frames"
    CATEGORY = "cspnodes"

    def generate_video_frames(self, prompt, num_inference_steps, height, width, num_frames):
        pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        video_frames = pipe(prompt, num_inference_steps=num_inference_steps, height=height, width=width, num_frames=num_frames).frames
        
        # If the frames are already numpy arrays, you can return them directly
        return (video_frames,)

NODE_CLASS_MAPPINGS = {
    "ImageDirIterator": ImageDirIterator,
    "Modelscopet2v": Modelscopet2v  # Added new node class
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDirIterator": "Image Dir Iterator",
    "Modelscopet2v": "Modelscopet2v"  # Added display name for new node
}