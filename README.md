# GPUstats
A Python library for getting the GPU status from NVIDA GPUs using nvidia-smi programmically in Python.

**Table of Contents**

1. [Installation](#installation)
2. [Usage](#usage)
3. [Examples](#examples)

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

```
GPUs = GPUstats.getGPUs()
```

```
deviceIDs = GPUstats.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5)
```

```
GPUavailability = GPUstats.getAvailability(GPUs, maxLoad = 0.5, maxMemory = 0.5)
```

```
deviceID = GPUstats.getFirstAvailable(maxLoad=0.5, maxMemory=0.5)
```

```
GPUstats.showUtilization()
```

(Work in progress)

## Examples

### Occupy only a single GPU in TensorFlow
By default, TensorFlow will occupy all available GPUs when using a gpu as a device (e.g. `tf.device('\gpu:0')`). By setting the environment variable `CUDA_VISIBLE_DEVICES`, the GPUs can be hidden from TensorFlow via CUDA (See [CUDA_VISIBLE_DEVICES - Masking GPUs](http://acceleware.com/blog/cudavisibledevices-masking-gpus)). Using GPUstats.py, the CUDA_VISIBLE_DEVICES can be set programmically based on the available GPUs.
Below is a minimum working example of how to only occupy 1 GPU in TensorFlow using GPUstats.
To run the code, copy it into a new python file (e.g. `demo_tensorflow_gpustats.py`) and run it (e.g. enter `python demo_tensorflow_gpustats.py` in a terminal).

```python
# Import os to set the environmental variable CUDA_VISIBLE_DEVICES
import os
import tensorflow as tf
import GPUstats

# Get the first available GPU
DEVICE_ID = GPUstats.getFirstAvailable()
DEVICE_ID = DEVICE_ID[0] # grab first element from list

# Set CUDA_VISIBLE_DEVICES to first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Set
device = '/gpu:' + str(DEVICE_ID)
print('Device: ' + device)

# Run a minimum working example on the selected GPU
with tf.Session() as sess:
    with tf.device(device):
        # Declare two numbers and add them together
        a = tf.constant(12)
        b = tf.constant(30)
        result = sess.run(a+b)
        print('a+b=' + str(result))

```

Your output should look something like the code block below. Notice how only one of the GPUs are found and created as a tensorflow device.

```
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Device: /gpu:0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:02:00.0
Total memory: 11.90GiB
Free memory: 11.76GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:02:00.0)
a+b=42

```
Comment the `os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)` line and compare the output. you output should look something like code block below. Notice, how all 4 GPUs are being found and created as a tensorflow device.

```
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
Device: /gpu:0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:02:00.0
Total memory: 11.90GiB
Free memory: 11.76GiB
W tensorflow/stream_executor/cuda/cuda_driver.cc:590] creating context when one is currently active; existing: 0x2c8e400
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 1 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:03:00.0
Total memory: 11.90GiB
Free memory: 11.76GiB
W tensorflow/stream_executor/cuda/cuda_driver.cc:590] creating context when one is currently active; existing: 0x2c92040
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 2 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:83:00.0
Total memory: 11.90GiB
Free memory: 11.76GiB
W tensorflow/stream_executor/cuda/cuda_driver.cc:590] creating context when one is currently active; existing: 0x2c95d90
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 3 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:84:00.0
Total memory: 11.90GiB
Free memory: 11.76GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 0 and 2
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 0 and 3
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 1 and 2
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 1 and 3
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 2 and 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 2 and 1
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 3 and 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:777] Peer access not supported between device ordinals 3 and 1
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0 1 2 3 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y Y N N 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 1:   Y Y N N 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 2:   N N Y Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 3:   N N Y Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:02:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:1) -> (device: 1, name: TITAN X (Pascal), pci bus id: 0000:03:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:2) -> (device: 2, name: TITAN X (Pascal), pci bus id: 0000:83:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:3) -> (device: 3, name: TITAN X (Pascal), pci bus id: 0000:84:00.0)
a+b=42
```
