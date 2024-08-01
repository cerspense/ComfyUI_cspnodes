# cspnodes
A ComfyUI node pack by cerspense

This package contains a collection of custom nodes for ComfyUI, designed to enhance your workflow with additional functionalities. Below is a detailed description of each node and its parameters.

## Table of Contents

1. [TextFileLineIterator](#textfilelineiterator)
2. [ImageDirIterator](#imagediriterator)
3. [VidDirIterator](#viddiriterator)
4. [Modelscopet2v](#modelscopet2v)
5. [Modelscopev2v](#modelscopev2v)
6. [SplitImageChannels](#splitimagechannels)
7. [RemapRange](#remaprange)
8. [ResizeByImage](#resizebyimage)
9. [IncrementEveryN](#incrementeveryn)
10. [DepthToNormalMap](#depthtormalmap)

## TextFileLineIterator

This node iterates through lines in a text file.

### Parameters:
- `file_path` (STRING): Path to the text file.
- `line_index` (INT, default: 0): Index of the line to retrieve.

### Output:
- (STRING): The selected line from the text file.

## ImageDirIterator

This node iterates through images in a directory or multiple directories, with various sorting and selection options.

### Parameters:
- `directory_path` (STRING): Path to the directory containing images.
- `glob_patterns` (STRING, default: "**/*.png, **/*.jpg"): Comma-separated list of glob patterns for selecting files.
- `image_index` (INT, default: 0): Starting index for image selection.
- `sort_by` (["date_modified", "name", "size", "random"]): Method to sort the images.
- `sort_order` (["ascending", "descending"]): Order of sorting.
- `batch_size` (INT, default: 1, min: 1, max: 64): Number of images to select at once.
- `increment_by_batch` (BOOLEAN, default: False): Whether to increment the index by batch size.
- `randomize_final_list` (BOOLEAN, default: False): Whether to randomize the final list of images.

### Output:
- (IMAGE): List of selected images.
- (STRING): List of filenames for the selected images.

## VidDirIterator

This node iterates through video files in a directory or multiple directories, with various sorting and selection options.

### Parameters:
- `directory_path` (STRING): Path to the directory containing video files.
- `glob_patterns` (STRING, default: "**/*.mp4, **/*.mov"): Comma-separated list of glob patterns for selecting files.
- `video_index` (INT, default: 0): Index of the video to select.
- `sort_by` (["date_modified", "name", "size", "random"]): Method to sort the videos.
- `sort_order` (["ascending", "descending"]): Order of sorting.
- `randomize_final_list` (BOOLEAN, default: False): Whether to randomize the final list of videos. This will allow full randomization across all specified folders and files. Otherwise, it will randomize within each folder before moving on to the next.

### Output:
- (STRING): Path to the selected video file.

## Modelscopet2v

This node generates video frames from text using the Modelscope text-to-video model.

### Parameters:
- `prompt` (STRING): Text prompt for video generation.
- `negative_prompt` (STRING, default: None): Negative prompt for generation.
- `model_path` (STRING, default: "cerspense/zeroscope_v2_576w"): Path to the model.
- `num_inference_steps` (INT, default: 25): Number of inference steps.
- `guidance_scale` (FLOAT, default: 9.0): Guidance scale for generation.
- `seed` (INT, default: 42): Seed for random generation.
- `width` (INT, default: 576): Width of the generated video.
- `height` (INT, default: 320): Height of the generated video.
- `num_frames` (INT, default: 24): Number of frames to generate.

### Output:
- (IMAGE): Generated video frames.

## Modelscopev2v

This node transforms video frames using the Modelscope video-to-video model.

### Parameters:
- `video_frames` (IMAGE): Input video frames.
- `prompt` (STRING): Text prompt for video transformation.
- `negative_prompt` (STRING, default: None): Negative prompt for transformation.
- `model_path` (STRING, default: "cerspense/zeroscope_v2_XL"): Path to the model.
- `strength` (FLOAT, default: 0.70): Strength of the transformation.
- `num_inference_steps` (INT, default: 25): Number of inference steps.
- `guidance_scale` (FLOAT, default: 8.50): Guidance scale for transformation.
- `seed` (INT, default: 42): Seed for random generation.
- `enable_forward_chunking` (BOOLEAN, default: False): Enable forward chunking for memory optimization.
- `enable_vae_slicing` (BOOLEAN, default: True): Enable VAE slicing for memory optimization.

### Output:
- (IMAGE): Transformed video frames.

## SplitImageChannels

This node splits an image into its red, green, and blue channels.

### Parameters:
- `image` (IMAGE): Input image to split.

### Output:
- (IMAGE): Red channel as a grayscale image.
- (IMAGE): Green channel as a grayscale image.
- (IMAGE): Blue channel as a grayscale image.

## RemapRange

This node remaps a value from one range to another.

### Parameters:
- `value` (FLOAT, default: 0.0, range: -10000.0 to 10000.0): Input value to remap.
- `input_min` (FLOAT, default: 0.0): Minimum of the input range.
- `input_max` (FLOAT, default: 1.0): Maximum of the input range.
- `output_min` (FLOAT, default: 0.0): Minimum of the output range.
- `output_max` (FLOAT, default: 1.0): Maximum of the output range.
- `clamp` (BOOLEAN, default: False): Whether to clamp the output to the output range.

### Output:
- (FLOAT): Remapped value.

## ResizeByImage

This node resizes an input image to match the dimensions of a reference image.

### Parameters:
- `reference_image` (IMAGE): Image to use as a reference for resizing.
- `input_image` (IMAGE, optional): Image to resize. If not provided, a black image is created.

### Output:
- (IMAGE): Resized image matching the dimensions of the reference image.

## IncrementEveryN

This node takes an input integer and outputs a new integer that increments every N steps of the input, with an optional offset.

### Parameters:
- `input_value` (INT, default: 0, min: 0): The input integer value that is incrementing.
- `step_size` (INT, default: 1, min: 1): The number of steps in the input required to increment the output by 1.
- `offset` (INT, default: 0): An integer value added to the final output to offset the result.

### Output:
- (INT): The incrementing output value.

### Behavior:
This node divides the input value by the step size, adds the offset, and returns the integer result. For example:
- If `step_size` is 6 and `offset` is 0:
  - Input values 0-5 will output 0
  - Input values 6-11 will output 1
  - Input values 12-17 will output 2
  - And so on...
- If `step_size` is 6 and `offset` is 10:
  - Input values 0-5 will output 10
  - Input values 6-11 will output 11
  - Input values 12-17 will output 12
  - And so on...

This node is useful for creating slower-changing values from rapidly incrementing inputs, which can be helpful in various animation and procedural generation scenarios. The offset parameter allows for further customization of the output range.


## DepthToNormalMap

This node converts depth maps to normal maps, with options to control intensity, axis flipping, and depth scaling. It's designed to work consistently across different image resolutions.

### Parameters:
- `depth_maps` (IMAGE): Input depth map image(s).
- `normal_intensity` (FLOAT, default: 1.0, range: 0.01 to 10.0): Intensity of the normal map effect. Higher values result in more pronounced normal maps.
- `flip_x` (BOOLEAN, default: True): Whether to flip the X-axis of the normal map.
- `flip_y` (BOOLEAN, default: False): Whether to flip the Y-axis of the normal map. Note that the Y-axis is flipped by default in the conversion process.
- `depth_scale` (FLOAT, default: 1.0, range: 0.02 to 2.0): Scaling factor for the depth values. Adjust this to match your depth map's range or to enhance subtle details.

### Output:
- (IMAGE): The generated normal map(s).

### Behavior:
This node takes depth map images as input and converts them to normal maps using a resolution-independent approach. The conversion process involves calculating gradients using Sobel filters, which provides better edge detection and is less sensitive to noise.

Key features and notes:
- The Y-axis of the normal map is flipped by default to match common normal map conventions. Use the `flip_y` option to un-flip it if necessary.
- The `depth_scale` parameter allows fine-tuning of the depth range. A value of 1.0 is equivalent to a 0.2 scaling in previous versions.
- The node uses a resolution-independent method, making it suitable for both low and high-resolution images.
- It can handle batches of images, processing multiple depth maps in a single operation.

Adjusting parameters:
- Start with `normal_intensity` at 1.0 and adjust as needed. Lower values give subtler normal maps, while higher values enhance details.
- Use `depth_scale` to adjust the perceived depth range. Increase it for depth maps with a narrow range of values, or decrease it if the depth variation is too extreme.
- For high-resolution images, lower `normal_intensity` values often work better due to more inherent detail.
- For low-resolution images, you might need to increase `normal_intensity` to make details more apparent.
