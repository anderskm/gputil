# GPUtil
`GPUtil` is a Python module for getting the GPU status from NVIDA GPUs using `nvidia-smi`.
`GPUtil` locates all GPUs on the computer, determines their availablity and returns a ordered list of available GPUs.
Availablity is based upon the current memory consumption and load of each GPU.
The module is written with GPU selection for Deep Learning in mind, but it is not task/library specific and it can be applied to any task, where it may be useful to identify available GPUs.

**Table of Contents**

1. [Requirements](#requirements)
1. [Installation](#installation)
1. [Usage](#usage)
   1. [Main functions](#main-functions)
   1. [Helper functions](#helper-functions)
1. [Examples](#examples)
   1. [Select first available GPU in Caffe](#select-first-available-gpu-in-caffe)
   1. [Occupy only 1 GPU in TensorFlow](#occupy-only-1-gpu-in-tensorflow)
   1. [Monitor GPU in a separate thread](#monitor-gpu-in-a-separate-thread)
1. [License](#license)

## Requirements
NVIDIA GPU with latest NVIDIA driver installed.
GPUtil uses the program `nvidia-smi` to get the GPU status of all available NVIDIA GPUs. `nvidia-smi` should be installed automatically, when you install your NVIDIA driver.

Supports both Python 2.X and 3.X.

Python libraries:
* subprocess ([The Python Standard Library](https://docs.python.org/3/library/subprocess.html))
* distutils ([The Python Standard Library](https://docs.python.org/3/library/distutils.html))
* math ([The Python Standard Library](https://docs.python.org/3/library/math.html))
* random ([The Python Standard Library](https://docs.python.org/3/library/random.html))
* time ([The Python Standard Library](https://docs.python.org/3/library/time.html))
* os ([The Python Standard Library](https://docs.python.org/3/library/os.html))
* sys ([The Python Standard Library](https://docs.python.org/3/library/sys.html))
* platform ([The Python Standard Library](https://docs.python.org/3/library/platform.html))

Tested on CUDA driver version 390.77 Python 2.7 and 3.5.

## Installation

1. Open a terminal (Ctrl+Shift+T)
2. Type `pip install gputil`
3. Test the installation
   1. Open a terminal in a folder other than the GPUtil folder
   2. Start a python console by typing `python` in the terminal
   3. In the newly opened python console, type:
      ```python
      import GPUtil
      GPUtil.showUtilization()
      ```
   4. Your output should look something like following, depending on your number of GPUs and their current usage:
      ```
       ID  GPU  MEM
      --------------
        0    0%   0%
      ```

### Old way of installation

1. Download or clone repository to your computer
2. Add GPUtil folder to ~/.bashrc
   1. Open a new terminal (Press Ctrl+Alt+T)
   2. Open bashrc:
      ```
      gedit ~/.bashrc
      ```
   3. Added your GPUtil folder to the environment variable `PYTHONPATH` (replace `<path_to_gputil>` with your folder path):
      ```
      export PYTHONPATH="$PYTHONPATH:<path_to_gputil>"

      Example:
      export PYTHONPATH="$PYTHONPATH:/home/anderskm/github/gputil"
      ```
   4. Save ~/.bashrc and close gedit
   5. Restart your terminal
1. Test the installation
   1. Open a terminal in a folder other than the GPUtil folder
   2. Start a python console by typing `python` in the terminal
   3. In the newly opened python console, type:
      ```python
      import GPUtil
      GPUtil.showUtilization()
      ```
   4. Your output should look something like following, depending on your number of GPUs and their current usage:
      ```
       ID  GPU  MEM
      --------------
        0    0%   0%
      ```

## Usage

To include `GPUtil` in your Python code, all you hve to do is included it at the beginning of your script:

```python
import GPUtil
```

Once included all functions are available. The functions along with a short description of inputs, outputs and their functionality can be found in the following two sections.

### Main functions

```python
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
```
Returns a list ids of available GPUs. Availablity is determined based on current memory usage and load. The order, maximum number of devices, their maximum load and maximum memory consumption are determined by the input arguments.

* Inputs
  * `order` - Deterimines the order in which the available GPU device ids are returned. `order` should be specified as one of the following strings:
    * `'first'` - orders available GPU device ids by ascending id (**defaut**)
    * `'last'` - orders available GPU device ids by descending id
    * `'random'` - orders the available GPU device ids randomly
    * `'load'`- orders the available GPU device ids by ascending load
    * `'memory'` - orders the available GPU device ids by ascending memory usage
  * `limit` - limits the number of GPU device ids returned to the specified number. Must be positive integer. (**default = 1**)
  * `maxLoad` - Maximum current relative load for a GPU to be considered available. GPUs with a load larger than `maxLoad` is not returned. (**default = 0.5**)
  * `maxMemory` - Maximum current relative memory usage for a GPU to be considered available. GPUs with a current memory usage larger than `maxMemory` is not returned. (**default = 0.5**)
  * `includeNan` - True/false flag indicating whether to include GPUs where either load or memory usage is NaN (indicating usage could not be retrieved). (**default = False**)
  * `excludeID` - List of IDs, which should be excluded from the list of available GPUs. See `GPU` class description. (**default = []**)
  * `excludeUUID` - Same as `excludeID` except it uses the UUID. (**default = []**)
* Outputs
  * deviceIDs - list of all available GPU device ids. A GPU is considered available, if the current load and memory usage is less than `maxLoad` and `maxMemory`, respectively. The list is ordered according to `order`. The maximum number of returned device ids is limited by `limit`.

```python
deviceID = GPUtil.getFirstAvailable(order = 'first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
```
Returns the first avaiable GPU. Availablity is determined based on current memory usage and load, and the ordering is determined by the specified order.
If no available GPU is found, an error is thrown.
When using the default values, it is the same as `getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5)`

* Inputs
  * `order` - See the description for `GPUtil.getAvailable(...)`
  * `maxLoad` - Maximum current relative load for a GPU to be considered available. GPUs with a load larger than `maxLoad` is not returned. (**default = 0.5**)
  * `maxMemory` - Maximum current relative memory usage for a GPU to be considered available. GPUs with a current memory usage larger than `maxMemory` is not returned. (**default = 0.5**)
  * `attempts` - Number of attempts the function should make before giving up finding an available GPU. (**default = 1**)
  * `interval` - Interval in seconds between each attempt to find an available GPU. (**default = 900** --> 15 mins)
  * `verbose` - If `True`, prints the attempt number before each attempt and the GPU id if an available is found.
  * `includeNan` - See the description for `GPUtil.getAvailable(...)`. (**default = False**)
  * `excludeID` - See the description for `GPUtil.getAvailable(...)`. (**default = []**)
  * `excludeUUID` - See the description for `GPUtil.getAvailable(...)`. (**default = []**)
* Outputs
  * deviceID - list with 1 element containing the first available GPU device ids. A GPU is considered available, if the current load and memory usage is less than `maxLoad` and `maxMemory`, respectively. The order and limit are fixed to `'first'` and `1`, respectively.


```python
GPUtil.showUtilization(all=False, attrList=None, useOldCode=False)
```
Prints the current status (id, memory usage, uuid load) of all GPUs
* Inputs
  * `all` - True/false flag indicating if all info on the GPUs should be shown. Overwrites `attrList`.
  * `attrList` - List of lists of `GPU` attributes to display. See code for more information/example.
  * `useOldCode` - True/false flag indicating if the old code to display GPU utilization should be used.
* Outputs
  * _None_

### Helper functions
```python
 class GPU
```
Helper class handle the attributes of each GPU. Quoted descriptions are copied from corresponding descriptions by `nvidia-smi`.
* Attributes for each `GPU`
  * `id` - "Zero based index of the GPU. Can change at each boot."
  * `uuid` - "This value is the globally unique immutable alphanumeric identifier of the GPU. It does not correspond to any physical label on the board. Does not change across reboots."
  * `load` - Relative GPU load. 0 to 1 (100%, full load). "Percent of time over the past sample period during which one or more kernels was executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product."
  * `memoryUtil` - Relative memory usage from 0 to 1 (100%, full usage). "Percent of time over the past sample period during which global (device) memory was being read or written. The sample period may be between 1 second and 1/6 second depending on the product."
  * `memoryTotal` - "Total installed GPU memory."
  * `memoryUsed` - "Total GPU memory allocated by active contexts."
  * `memoryFree` - "Total free GPU memory."
  * `driver` - "The version of the installed NVIDIA display driver."
  * `name` - "The official product name of the GPU."
  * `serial` - This number matches the serial number physically printed on each board. It is a globally unique immutable alphanumeric value.
  * `display_mode` - "A flag that indicates whether a physical display (e.g. monitor) is currently connected to any of the GPU's connectors. "Enabled" indicates an attached display. "Disabled" indicates otherwise."
  * `display_active` - "A flag that indicates whether a display is initialized on the GPU's (e.g. memory is allocated on the device for display). Display can be active even when no monitor is physically attached. "Enabled" indicates an active display. "Disabled" indicates otherwise."

```python
GPUs = GPUtil.getGPUs()
```
* Inputs
  * _None_
* Outputs
  * `GPUs` - list of all GPUs. Each `GPU` corresponds to one GPU in the computer and contains a device id, relative load and relative memory usage.

```python
GPUavailability = GPUtil.getAvailability(GPUs, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
```
Given a list of `GPUs` (see `GPUtil.getGPUs()`), return a equally sized list of ones and zeroes indicating which corresponding GPUs are available.

* Inputs
  * `GPUs` - List of `GPUs`. See `GPUtil.getGPUs()`
  * `maxLoad` - Maximum current relative load for a GPU to be considered available. GPUs with a load larger than `maxLoad` is not returned. (**default = 0.5**)
  * `maxMemory` - Maximum current relative memory usage for a GPU to be considered available. GPUs with a current memory usage larger than `maxMemory` is not returned. (**default = 0.5**)
  * `includeNan` - See the description for `GPUtil.getAvailable(...)`. (**default = False**)
  * `excludeID` - See the description for `GPUtil.getAvailable(...)`. (**default = []**)
  * `excludeUUID` - See the description for `GPUtil.getAvailable(...)`. (**default = []**)
* Outputs
  * GPUavailability - binary list indicating if `GPUs` are available or not. A GPU is considered available, if the current load and memory usage is less than `maxLoad` and `maxMemory`, respectively.


See [demo_GPUtil.py](https://github.com/anderskm/gputil/blob/master/demo_GPUtil.py) for examples and more details.

## Examples


### Select first available GPU in Caffe
In the Deep Learning library [Caffe](http://caffe.berkeleyvision.org/), the user can switch between using the CPU or GPU through their Python interface.
This is done by calling the methods `caffe.set_mode_cpu()` and `caffe.set_mode_gpu()`, respectively.
Below is a minimum working example for selecting the first available GPU with GPUtil to run a Caffe network.

```python
# Import caffe and GPUtil
import caffe
import GPUtil

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

# Select GPU mode
caffe.set_mode_gpu()
# Select GPU id
caffe.set_device(DEVICE_ID)

# Initialize your network here

```

**Note:** At the time of writing this example, the Caffe Python wrapper only supports 1 GPU, although the underlying code supports multiple GPUs.
Calling directly Caffe from the terminal allows for using multiple GPUs.

### Occupy only 1 GPU in TensorFlow
By default, [TensorFlow](https://www.tensorflow.org/)  will occupy all available GPUs when using a gpu as a device (e.g. `tf.device('\gpu:0')`).
By setting the environment variable `CUDA_VISIBLE_DEVICES`, the user can mask which GPUs should be visible to TensorFlow via CUDA (See [CUDA_VISIBLE_DEVICES - Masking GPUs](http://acceleware.com/blog/cudavisibledevices-masking-gpus)). Using GPUtil.py, the CUDA_VISIBLE_DEVICES can be set programmatically based on the available GPUs.
Below is a minimum working example of how to occupy only 1 GPU in TensorFlow using GPUtil.
To run the code, copy it into a new python file (e.g. `demo_tensorflow_gputil.py`) and run it (e.g. enter `python demo_tensorflow_gputil.py` in a terminal).

**Note:** Even if you set the device you run your code on to a CPU, TensorFlow will occupy all available GPUs. To avoid this, all GPUs can be hidden from TensorFlow with `os.environ["CUDA_VISIBLE_DEVICES"] = ''`.

```python
# Import os to set the environment variable CUDA_VISIBLE_DEVICES
import os
import tensorflow as tf
import GPUtil

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

# Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0
device = '/gpu:0'
print('Device ID (unmasked): ' + str(DEVICE_ID))
print('Device ID (masked): ' + str(0))

# Run a minimum working example on the selected GPU
# Start a session
with tf.Session() as sess:
    # Select the device
    with tf.device(device):
        # Declare two numbers and add them together in TensorFlow
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
Comment the `os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)` line and compare the two outputs.
Depending on your number of GPUs, your output should look something like code block below.
Notice, how all 4 GPUs are being found and created as a tensorflow device, whereas when `CUDA_VISIBLE_DEVICES` was set, only 1 GPU was found and created.

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

### Monitor GPU in a separate thread
If using GPUtil to monitor GPUs during training, it may show 0% utilization. A way around this is to use a separate monitoring thread.
```python
import GPUtil
from threading import Thread
import time

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        
# Instantiate monitor with a 10-second delay between updates
monitor = Monitor(10)

# Train, etc.

# Close monitor
monitor.stop()
```

## License
See [LICENSE](https://github.com/anderskm/gputil/blob/master/LICENSE.txt)
