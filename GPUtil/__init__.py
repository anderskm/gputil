# GPUtil - GPU utilization
# 
# A Python module for programmically getting the GPU utilization from NVIDA GPUs using nvidia-smi
#
# Author: Anders Krogh Mortensen (anderskm)
# Date:   16 January 2017
# Web:    https://github.com/anderskm/gputil
#
# LICENSE
#
# MIT License
#
# Copyright (c) 2017 anderskm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from subprocess import Popen, PIPE, STDOUT
import os
import numpy as np
import random
import time

class  GPU:
    def __init__(self, ID, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active):
        self.id = ID
        self.load = load
        self.memoryUtil = float(memoryUsed)/float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active

def getGPUs():
    # Get ID, processing and memory utilization for all GPUs
    p = Popen(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode","--format=csv,noheader,nounits"],stdout=PIPE)
    output = p.stdout.read().decode('UTF-8')
    # output = output[2:-1] # Remove b' and ' from string added by python
    #print(output)
    ## Parse output
    # Split on line break
    #lines = output.split('\n')
    lines = output.split(os.linesep)
    #print(lines)
    numDevices = len(lines)-1
    deviceIds = np.empty(numDevices,dtype=int)
    gpuUtil = np.empty(numDevices,dtype=float)
    memTotal = np.empty(numDevices,dtype=int)
    memUsed = np.empty(numDevices,dtype=int)
    memFree = np.empty(numDevices,dtype=int)
    driver = []
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        #print(line)
        vals = line.split(', ')
        #print(vals)
        for i in range(10):
#            print(vals[i])
            if (i == 0):
                deviceIds[g] = int(vals[i])
            elif (i == 1):
                gpuUtil[g] = float(vals[i])/100
            elif (i == 2):
                memTotal[g] = int(vals[i])
            elif (i == 3):
                memUsed[g] = int(vals[i])
            elif (i == 4):
                memFree[g] = int(vals[i])
            elif (i == 5):
                driver = vals[i]
            elif (i == 6):
                gpu_name = vals[i]
            elif (i == 7):
                serial = vals[i]
            elif (i == 8):
                display_active = vals[i]
            elif (i == 9):
                display_mode = vals[i]
        GPUs.append(GPU(deviceIds[g], gpuUtil[g], memTotal[g], memUsed[g], memFree[g], driver, gpu_name,serial,display_mode, display_active))
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
        GPUs.sort(key=lambda x: x.memoryUtil, reverse=False)

    # Extract the number of desired GPUs, but limited to the total number of available GPUs
    GPUs = GPUs[0:np.minimum(limit, len(GPUs))]

    # Extract the device IDs from the GPUs and return them
    deviceIds = [GPUs[g].id for g in range(len(GPUs))]
    return deviceIds

def getAvailability(GPUs, maxLoad = 0.5, maxMemory = 0.5):
    # Determine, which GPUs are available
    GPUavailability = np.zeros(len(GPUs))
    for i in range(len(GPUs)):
        if (GPUs[i].load < maxLoad) & (GPUs[i].memoryUtil < maxMemory):
            GPUavailability[i] = 1
    return GPUavailability

def getFirstAvailable(order = 'first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False):
    #GPUs = getGPUs()
    #firstAvailableGPU = np.NaN
    #for i in range(len(GPUs)):
    #    if (GPUs[i].load < maxLoad) & (GPUs[i].memory < maxMemory):
    #        firstAvailableGPU = GPUs[i].id
    #        break
    #return firstAvailableGPU       
    for i in range(attempts):
        if (verbose):
            print('Attempting (' + str(i+1) + '/' + str(attempts) + ') to locate available GPU.')
        # Get first available GPU
        available = getAvailable(order = order, limit = 1, maxLoad = maxLoad, maxMemory = maxMemory)
        # If an available GPU was found, break for loop.
        if (available):
            if (verbose):
                print('GPU ' + str(available) + ' located!')
            break
        # If this is not the last attempt, sleep for 'interval' seconds
        if (i != attempts-1):
            time.sleep(interval)
    # Check if an GPU was found, or if the attempts simply ran out. Throw error, if no GPU was found
    if (not(available)):
        raise RuntimeError('Could not find an available GPU after ' + str(attempts) + ' attempts with ' + str(interval) + ' seconds interval.')
    
    # Return found GPU
    return available

def showUtilization(all=False):
    GPUs = getGPUs()
    if (all):
        print(' ID | Name | Serial || GPU util. | Memory util. || Memory total | Memory used | Memory free || Display mode | Display active |')
        print('------------------------------------------------------------------------------------------------------------------------------')
        for i in range(len(GPUs)):
            print(' {0:2d} | {1:s}  | {2:s} || {3:3.0f}% | {4:3.0f}% || {5:d}MB | {6:d}MB | {7:d}MB || {8:s} | {9:s}'.format(GPUs[i].id,GPUs[i].name,GPUs[i].serial,GPUs[i].load*100,GPUs[i].memoryUtil*100,GPUs[i].memoryTotal,GPUs[i].memoryUsed,GPUs[i].memoryFree,GPUs[i].display_mode,GPUs[i].display_active))
    else:
        print(' ID  GPU  MEM')
        print('--------------')
        for i in range(len(GPUs)):
            print(' {0:2d} {1:3.0f}% {2:3.0f}%'.format(GPUs[i].id,GPUs[i].load*100,GPUs[i].memoryUtil*100))
