# GPUstats
A Python library for getting the GPU status from NVIDA GPUs using nvidia-smi programmically in Python.

**Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)

## Installation
1. Download or clone repository to your computer
2. Add folder to ~/.bashrc
  1. Open a new terminal (Press Ctrl+Alt+T)
  2. Open bashrc:

      ```
      gedit ~/.bashrc
      ```

  3. Added your GPUstats folder to PYTHONPATH (replace <path_to_gpustats> with your folder path):

      ```
      export PYTHONPATH="$PYTHONPATH:<path_to_gpustats>"

      Example:
      export PYTHONPATH="$PYTHONPATH:/home/anderskm/github/gpustats"
      ```

  4. Save ~/.bashrc and close gedit
  5. Restart your terminal

## Usage
See [demo_GPUstats.py](https://github.com/anderskm/gpustats/blob/master/demo_GPUstats.py)

(Work in progress)

### Occupy only a single GPU in TensorFlow
By default, TensorFlow will occupy all available GPUs when using a gpu as a device (e.g. `tf.device('\gpu:0')`). By setting the environment variable `CUDA_VISIBLE_DEVICES`, the GPUs can be hidden from TensorFlow via CUDA (See [CUDA_VISIBLE_DEVICES - Masking GPUs](http://acceleware.com/blog/cudavisibledevices-masking-gpus)). Using GPUstats.py, the CUDA_VISIBLE_DEVICES can be set programmically based on the available GPUs.

```python
# Import os to set the environmental variable CUDA_VISIBLE_DEVICES
import os
import tensorflow as tf
import GPUstats

# Get the first available GPU
DEVICE_ID = GPUstats.getFirstAvailable()

# Set CUDA_VISIBLE_DEVICES to first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Set
device = '/gpu:' + str(DEVICE_ID)
print('Device: ' + device)

with tf.session() as sess:
    with tf.device(device):
        a = tf.constant(12)
        b = tf.constant(30)

        print(sess.run(a+b))

```
