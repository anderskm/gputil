from subprocess import Popen, PIPE, STDOUT
import numpy as np
import random

class  GPU:
    def __init__(self, ID, load, memory):
        self.id = ID
        self.load = load
        self.memory = memory

def getGPUs():
    # Get ID, processing and memory utilization for all GPUs
    p = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,utilization.memory","--format=csv,noheader,nounits"],stdout=PIPE)
    output = str(p.stdout.read())

    ## Parse output
    # Split on line break
    lines = output.split('\n')
    numDevices = len(lines)-1
    deviceIds = np.empty(numDevices,dtype=int)
    gpuUtil = np.empty(numDevices,dtype=float)
    memUtil = np.empty(numDevices,dtype=float)
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
#        print(line)
        vals = line.split(', ')
#        print(vals)
        for i in range(3):
#            print(vals[i])
            if (i == 0):
                deviceIds[g] = int(vals[i])
            elif (i == 1):
                gpuUtil[g] = float(vals[i])/100
            elif (i == 2):
                memUtil[g] = float(vals[i])/100
        GPUs.append(GPU(deviceIds[g], gpuUtil[g], memUtil[g]))
    return GPUs #(deviceIds, gpuUtil, memUtil)

def getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5):
    # order = first | last | random | load | memory
    #    first --> select the GPU with the lowest ID (DEFAULT)
    #    last --> select the GPU with the highest ID
    #    random --> select a random available GPU
    #    lowest --> select the GPU with the lowest load
    # limit = 1 (DEFAULT), 2, ..., Inf
    #     Limit sets the upper limit for the number of GPUs to return. E.g. if limit = 2, but only one is available, only one is returned.

    # Get devise IDs, load and memory usage
    GPUs = getGPUs()

    # Determine, which GPUs are available
    GPUavailability = np.array(getAvailability(GPUs, maxLoad, maxMemory))
    availAbleGPUindex = np.where(GPUavailability == 1)[0]
    # Discard unavailable GPUs
    GPUs = [GPUs[g] for g in availAbleGPUindex]

    # Sort available GPUs according to the order argument
    if (order == 'first'):
        GPUs.sort(key=lambda x: x.id, reverse=False)
    elif (order == 'last'):
        GPUs.sort(key=lambda x: x.id, reverse=True)
    elif (order == 'random'):
        GPUs = [GPUs[g] for g in random.sample(range(0,len(GPUs)),len(GPUs))]
    elif (order == 'load'):
        GPUs.sort(key=lambda x: x.load, reverse=False)
    elif (order == 'memory'):
        GPUs.sort(key=lambda x: x.memory, reverse=False)

    # Extract the number of desired GPUs, but limited to the total number of available GPUs
    GPUs = GPUs[0:np.minimum(limit, len(GPUs))]

    # Extract the device IDs from the GPUs and return them
    deviceIds = [GPUs[g].id for g in range(len(GPUs))]
    return deviceIds

def getAvailability(GPUs, maxLoad = 0.5, maxMemory = 0.5):
    # Determine, which GPUs are available
    GPUavailability = np.zeros(len(GPUs))
    for i in range(len(GPUs)):
        if (GPUs[i].load < maxLoad) & (GPUs[i].memory < maxMemory):
            GPUavailability[i] = 1
    return GPUavailability

def getFirstAvailable(maxLoad=0.5, maxMemory=0.5):
    #GPUs = getGPUs()
    #firstAvailableGPU = np.NaN
    #for i in range(len(GPUs)):
    #    if (GPUs[i].load < maxLoad) & (GPUs[i].memory < maxMemory):
    #        firstAvailableGPU = GPUs[i].id
    #        break
    #return firstAvailableGPU
    return getAvailable(order = 'first', limit = 1, maxLoad = maxLoad, maxMemory = maxMemory)

def showUtilization():
    GPUs = getGPUs()
    print(' ID  GPU  MEM')
    print('--------------')
    for i in range(len(GPUs)):
        print(' {0:2d}  {1:3.0f}% {2:3.0f}%'.format(GPUs[i].id,GPUs[i].load*100,GPUs[i].memory*100))
