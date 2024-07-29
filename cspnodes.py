import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image, ImageOps
import numpy as np
import random
import torch.nn.functional as F
import glob


class TextFileLineIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "line_index": ("INT", {"default": 0})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_line_by_index"
    CATEGORY = "cspnodes"

    def get_line_by_index(self, file_path, line_index):
        # Read all lines from the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Wrap the index around using modulo
        line_index = line_index % len(lines)

        # Get the specified line and strip any surrounding whitespace
        line = lines[line_index].strip()

        return (line,)


class ImageDirIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {}),
                "glob_patterns": ("STRING", {"default": "**/*.png, **/*.jpg"}),
                "image_index": ("INT", {"default": 0}),
                "sort_by": (["date_modified", "name", "size", "random"],),
                "sort_order": (["ascending", "descending"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "increment_by_batch": ("BOOLEAN", {"default": False}),
                "randomize_final_list": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "get_images_by_index"
    CATEGORY = "cspnodes"
    OUTPUT_IS_LIST = (True, True)

    def get_images_by_index(self, directory_path, glob_patterns, image_index, sort_by, sort_order, batch_size, increment_by_batch, randomize_final_list):
        # Split and clean the glob patterns
        patterns = [p.strip() for p in glob_patterns.split(',') if p.strip()]
        
        # Get list of image files including subdirectories for all patterns
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(directory_path, pattern), recursive=True))
        
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]

        if len(image_files) == 0:
            raise FileNotFoundError(f"No valid image files found in directory '{directory_path}' with patterns '{glob_patterns}'.")

        # Define sorting key functions
        sort_functions = {
            "date_modified": lambda x: os.path.getmtime(x),
            "name": lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0),
            "size": lambda x: os.path.getsize(x),
            "random": lambda x: random.random(),
        }

        # Group files by subdirectory
        subdirs = {}
        for file in image_files:
            subdir = os.path.dirname(file)
            if subdir not in subdirs:
                subdirs[subdir] = []
            subdirs[subdir].append(file)

        # Sort files within each subdirectory
        sorted_files = []
        for subdir, files in subdirs.items():
            if sort_by == "random":
                random.shuffle(files)
            else:
                files.sort(key=sort_functions[sort_by], reverse=(sort_order == "descending"))
            sorted_files.extend(files)

        # Randomize the entire list if requested
        if randomize_final_list:
            random.shuffle(sorted_files)

        # Calculate the starting index based on the increment_by_batch option
        start_index = image_index * batch_size if increment_by_batch else image_index

        # Wrap the index around using modulo
        start_index = start_index % len(sorted_files)

        # Select the batch of images
        selected_files = [sorted_files[(start_index + i) % len(sorted_files)] for i in range(batch_size)]

        # Load and preprocess the images
        images = []
        filenames = []
        for file in selected_files:
            try:
                i = Image.open(file)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                images.append(image)
                
                filename = os.path.splitext(os.path.basename(file))[0]
                filename = filename.encode('utf-8').decode('unicode_escape')
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading image {file}: {str(e)}")

        return (images, filenames)


class VidDirIterator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {}),
                "glob_patterns": ("STRING", {"default": "**/*.mp4, **/*.mov"}),
                "video_index": ("INT", {"default": 0}),
                "sort_by": (["date_modified", "name", "size", "random"],),
                "sort_order": (["ascending", "descending"],),
                "randomize_final_list": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_video_path_by_index"
    CATEGORY = "cspnodes"

    def get_video_path_by_index(self, directory_path, glob_patterns, video_index, sort_by, sort_order, randomize_final_list):
        # Split and clean the glob patterns
        patterns = [p.strip() for p in glob_patterns.split(',') if p.strip()]
        
        # Get list of video files including subdirectories for all patterns
        video_files = []
        for pattern in patterns:
            video_files.extend(glob.glob(os.path.join(directory_path, pattern), recursive=True))
        
        video_files = [f for f in video_files if f.lower().endswith(('.mov', '.mp4'))]

        if len(video_files) == 0:
            raise FileNotFoundError(f"No valid video files found in directory '{directory_path}' with patterns '{glob_patterns}'.")

        # Define sorting key functions
        sort_functions = {
            "date_modified": lambda x: os.path.getmtime(x),
            "name": lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0),
            "size": lambda x: os.path.getsize(x),
            "random": lambda x: random.random(),
        }

        # Sort the video files
        if sort_by == "random":
            random.shuffle(video_files)
        else:
            video_files.sort(key=sort_functions[sort_by], reverse=(sort_order == "descending"))

        # Randomize the entire list if requested
        if randomize_final_list:
            random.shuffle(video_files)

        # Wrap the index around using modulo
        video_index = video_index % len(video_files)

        # Return the video file path as a string
        return (video_files[video_index],)

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

        # Normalize the tensor to have values between 0 and 1 if they are in the range 0-255
        if video_frames.max() > 1.0:
            video_frames = video_frames / 255.0

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

        # Normalize the tensor to have values between 0 and 1 if they are in the range 0-255
        if video_frames.max() > 1.0:
            video_frames = video_frames / 255.0
        
        # The expected shape is (num_frames, height, width, channels)
        video_frames = video_frames.squeeze(0).permute(0, 1, 2, 3)

        # Convert the tensor to CPU and to uint8 if it's not already
        video_frames = video_frames.to('cpu')

        # return (video_frames_numpy,)
        return (video_frames,)
    
class SplitImageChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    FUNCTION = "split_channels"
    CATEGORY = "cspnodes"

    def split_channels(self, image):
        # Split the image into red, green, and blue channels
        red_channel = image[:, :, :, 0]
        green_channel = image[:, :, :, 1]
        blue_channel = image[:, :, :, 2]

        # Convert each channel to a black and white image
        red_bw = torch.stack([red_channel, red_channel, red_channel], dim=-1)
        green_bw = torch.stack([green_channel, green_channel, green_channel], dim=-1)
        blue_bw = torch.stack([blue_channel, blue_channel, blue_channel], dim=-1)

        return red_bw, green_bw, blue_bw
    
class RemapRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "input_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "input_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "output_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "output_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                "clamp": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "remap_value"
    CATEGORY = "cspnodes"

    def remap_value(self, value, input_min, input_max, output_min, output_max, clamp):
        # Calculate the input and output ranges
        input_range = input_max - input_min
        output_range = output_max - output_min
        
        # Perform the remapping
        if input_range == 0:
            remapped = output_min
        else:
            remapped = ((value - input_min) / input_range) * output_range + output_min
        
        # Clamp the output if requested
        if clamp:
            remapped = max(min(remapped, output_max), output_min)
        
        return (remapped,)

class ResizeByImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
            },
            "optional": {
                "input_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "cspnodes"

    def resize_image(self, reference_image, input_image=None):
        # Get the dimensions of the reference image
        _, height, width, _ = reference_image.shape

        if input_image is None:
            # Create a black image batch if no input is provided
            resized_image = torch.zeros_like(reference_image)
        else:
            # Resize the input image batch to match the reference image dimensions
            # Convert from (batch, height, width, channels) to (batch, channels, height, width)
            input_image = input_image.permute(0, 3, 1, 2)
            
            # Perform the resize operation on the entire batch
            resized_image = F.interpolate(input_image, size=(height, width), mode='bilinear', align_corners=False)
            
            # Convert back to (batch, height, width, channels)
            resized_image = resized_image.permute(0, 2, 3, 1)

        return (resized_image,)
    
class IncrementEveryN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": ("INT", {"default": 0, "min": 0, "step": 1}),
                "step_size": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "increment_every_n"
    CATEGORY = "cspnodes"

    def increment_every_n(self, input_value, step_size):
        output_value = input_value // step_size
        return (output_value,)

    
NODE_CLASS_MAPPINGS = {
    "IncrementEveryN": IncrementEveryN,
    "ResizeByImage": ResizeByImage,
    "SplitImageChannels": SplitImageChannels,
    "RemapRange": RemapRange,
    "TextFileLineIterator": TextFileLineIterator,
    "ImageDirIterator": ImageDirIterator,
    "VidDirIterator": VidDirIterator,
    "Modelscopet2v": Modelscopet2v,
    "Modelscopev2v": Modelscopev2v,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IncrementEveryN": "Increment Every N",
    "ResizeByImage": "Resize By Image",
    "SplitImageChannels": "Split Image Channels",
    "RemapRange": "Remap Range",
    "TextFileLineIterator": "Text File Line Iterator",
    "ImageDirIterator": "Image Dir Iterator",
    "VidDirIterator": "Vid Dir Iterator",
    "Modelscopet2v": "Modelscope t2v",
    "Modelscopev2v": "Modelscope v2v",
}
