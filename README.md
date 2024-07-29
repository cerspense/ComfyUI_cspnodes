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

...

[Previous content remains unchanged]

...

## IncrementEveryN

This node takes an input integer and outputs a new integer that increments every N steps of the input.

### Parameters:
- `input_value` (INT, default: 0, min: 0): The input integer value that is incrementing.
- `step_size` (INT, default: 1, min: 1): The number of steps in the input required to increment the output by 1.

### Output:
- (INT): The incrementing output value.

### Behavior:
This node divides the input value by the step size and returns the integer result. For example:
- If `step_size` is 6:
  - Input values 0-5 will output 0
  - Input values 6-11 will output 1
  - Input values 12-17 will output 2
  - And so on...

This node is useful for creating slower-changing values from rapidly incrementing inputs, which can be helpful in various animation and procedural generation scenarios.
