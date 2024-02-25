import os
import torch
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

NODE_CLASS_MAPPINGS = {
    "ImageDirIterator": ImageDirIterator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDirIterator": "Image Dir Iterator"
}