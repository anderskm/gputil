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

from distutils import spawn
import os
import math
import random
import time
import sys
import platform
import subprocess


__version__ = '1.4.0'

class GPU:
    def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
        self.id = ID
        self.uuid = uuid
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
        self.temperature = temp_gpu

    def __str__(self):
        return str(self.__dict__)


class GPUProcess:
    def __init__(self, pid, processName, gpuId, gpuUuid, gpuName, usedMemory,
                 uid, uname):
        self.pid = pid
        self.processName = processName
        self.gpuId = gpuId
        self.gpuUuid = gpuUuid
        self.gpuName = gpuName
        self.usedMemory = usedMemory
        self.uid = uid
        self.uname = uname

    def __str__(self):
        return str(self.__dict__)

def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float('nan')
    return number

def getNvidiaSmiCmd():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi 
        # could not be found from the environment path, 
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"
    return nvidia_smi
	

def getGPUs():
    # Get ID, processing and memory utilization for all GPUs
    nvidia_smi = getNvidiaSmiCmd()
    try:
        p = subprocess.run([
            nvidia_smi,
            "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], stdout=subprocess.PIPE, encoding='utf8')
        stdout, stderror = p.stdout, p.stderr
    except:
        return []
    output = stdout
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    numDevices = len(lines)-1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        #print(line)
        vals = line.split(', ')
        #print(vals)
        deviceIds = int(vals[0])
        uuid = vals[1]
        gpuUtil = safeFloatCast(vals[2]) / 100
        memTotal = safeFloatCast(vals[3])
        memUsed = safeFloatCast(vals[4])
        memFree = safeFloatCast(vals[5])
        driver = vals[6]
        gpu_name = vals[7]
        serial = vals[8]
        display_active = vals[9]
        display_mode = vals[10]
        temp_gpu = safeFloatCast(vals[11]);
        GPUs.append(GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu))
    return GPUs  # (deviceIds, gpuUtil, memUtil)


def getGPUProcesses():
    """Get all gpu compute processes."""

    global gpuUuidToIdMap

    nvidia_smi = getNvidiaSmiCmd()
    try:
        p = subprocess.run([
            nvidia_smi,
            "--query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory",
            "--format=csv,noheader,nounits"
        ], stdout=subprocess.PIPE, encoding='utf8')
        stdout, stderror = p.stdout, p.stderr
    except:
        return []
    output = stdout
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    numProcesses = len(lines) - 1
    processes = []
    for g in range(numProcesses):
        line = lines[g]
        #print(line)
        vals = line.split(', ')
        #print(vals)
        pid = int(vals[0])
        processName = vals[1]
        gpuUuid = vals[2]
        gpuName = vals[3]
        usedMemory = safeFloatCast(vals[4])
        gpuId = gpuUuidToIdMap[gpuUuid]
        if gpuId is None:
            gpuId = -1

        # get uid and uname owner of the pid
        try:
            p = subprocess.run(['ps', f'-p{pid}', '-oruid=,ruser='],
                               stdout=subprocess.PIPE, encoding='utf8')
            uid, uname = p.stdout.split()
            uid = int(uid)
        except:
            uid, uname = -1, ''

        processes.append(GPUProcess(pid, processName, gpuId, gpuUuid,
                                    gpuName, usedMemory, uid, uname))
    return processes


def getAvailable(order = 'first', limit=1, maxLoad=0.5, maxMemory=0.5, memoryFree=0, includeNan=False, excludeID=[], excludeUUID=[]):
    # order = first | last | random | load | memory
    #    first --> select the GPU with the lowest ID (DEFAULT)
    #    last --> select the GPU with the highest ID
    #    random --> select a random available GPU
    #    load --> select the GPU with the lowest load
    #    memory --> select the GPU with the most memory available
    # limit = 1 (DEFAULT), 2, ..., Inf
    #     Limit sets the upper limit for the number of GPUs to return. E.g. if limit = 2, but only one is available, only one is returned.

    # Get device IDs, load and memory usage
    GPUs = getGPUs()

    # Determine, which GPUs are available
    GPUavailability = getAvailability(GPUs, maxLoad=maxLoad, maxMemory=maxMemory, memoryFree=memoryFree, includeNan=includeNan, excludeID=excludeID, excludeUUID=excludeUUID)
    availAbleGPUindex = [idx for idx in range(0,len(GPUavailability)) if (GPUavailability[idx] == 1)]
    # Discard unavailable GPUs
    GPUs = [GPUs[g] for g in availAbleGPUindex]

    # Sort available GPUs according to the order argument
    if (order == 'first'):
        GPUs.sort(key=lambda x: float('inf') if math.isnan(x.id) else x.id, reverse=False)
    elif (order == 'last'):
        GPUs.sort(key=lambda x: float('-inf') if math.isnan(x.id) else x.id, reverse=True)
    elif (order == 'random'):
        GPUs = [GPUs[g] for g in random.sample(range(0,len(GPUs)),len(GPUs))]
    elif (order == 'load'):
        GPUs.sort(key=lambda x: float('inf') if math.isnan(x.load) else x.load, reverse=False)
    elif (order == 'memory'):
        GPUs.sort(key=lambda x: float('inf') if math.isnan(x.memoryUtil) else x.memoryUtil, reverse=False)

    # Extract the number of desired GPUs, but limited to the total number of available GPUs
    GPUs = GPUs[0:min(limit, len(GPUs))]

    # Extract the device IDs from the GPUs and return them
    deviceIds = [gpu.id for gpu in GPUs]

    return deviceIds

#def getAvailability(GPUs, maxLoad = 0.5, maxMemory = 0.5, includeNan = False):
#    # Determine, which GPUs are available
#    GPUavailability = np.zeros(len(GPUs))
#    for i in range(len(GPUs)):
#        if (GPUs[i].load < maxLoad or (includeNan and np.isnan(GPUs[i].load))) and (GPUs[i].memoryUtil < maxMemory  or (includeNan and np.isnan(GPUs[i].memoryUtil))):
#            GPUavailability[i] = 1

def getAvailability(GPUs, maxLoad=0.5, maxMemory=0.5, memoryFree=0, includeNan=False, excludeID=[], excludeUUID=[]):
    # Determine, which GPUs are available
    GPUavailability = [1 if (gpu.memoryFree>=memoryFree) and (gpu.load < maxLoad or (includeNan and math.isnan(gpu.load))) and (gpu.memoryUtil < maxMemory  or (includeNan and math.isnan(gpu.memoryUtil))) and ((gpu.id not in excludeID) and (gpu.uuid not in excludeUUID)) else 0 for gpu in GPUs]
    return GPUavailability

def getFirstAvailable(order = 'first', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False, includeNan=False, excludeID=[], excludeUUID=[]):
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
        available = getAvailable(order=order, limit=1, maxLoad=maxLoad, maxMemory=maxMemory, includeNan=includeNan, excludeID=excludeID, excludeUUID=excludeUUID)
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


def showUtilization(all=False, attrList=None, useOldCode=False):
    GPUs = getGPUs()
    if (all):
        if (useOldCode):
            print(' ID | Name | Serial | UUID || GPU util. | Memory util. || Memory total | Memory used | Memory free || Display mode | Display active |')
            print('------------------------------------------------------------------------------------------------------------------------------')
            for gpu in GPUs:
                print(' {0:2d} | {1:s}  | {2:s} | {3:s} || {4:3.0f}% | {5:3.0f}% || {6:.0f}MB | {7:.0f}MB | {8:.0f}MB || {9:s} | {10:s}'.format(gpu.id,gpu.name,gpu.serial,gpu.uuid,gpu.load*100,gpu.memoryUtil*100,gpu.memoryTotal,gpu.memoryUsed,gpu.memoryFree,gpu.display_mode,gpu.display_active))
        else:
            attrList = [[{'attr':'id','name':'ID'},
                         {'attr':'name','name':'Name'},
                         {'attr':'serial','name':'Serial'},
                         {'attr':'uuid','name':'UUID'}],
                        [{'attr':'temperature','name':'GPU temp.','suffix':'C','transform': lambda x: x,'precision':0},
						 {'attr':'load','name':'GPU util.','suffix':'%','transform': lambda x: x*100,'precision':0},
                         {'attr':'memoryUtil','name':'Memory util.','suffix':'%','transform': lambda x: x*100,'precision':0}],
                        [{'attr':'memoryTotal','name':'Memory total','suffix':'MB','precision':0},
                         {'attr':'memoryUsed','name':'Memory used','suffix':'MB','precision':0},
                         {'attr':'memoryFree','name':'Memory free','suffix':'MB','precision':0}],
                        [{'attr':'display_mode','name':'Display mode'},
                         {'attr':'display_active','name':'Display active'}]]
        
    else:
        if (useOldCode):
            print(' ID  GPU  MEM')
            print('--------------')
            for gpu in GPUs:
                print(' {0:2d} {1:3.0f}% {2:3.0f}%'.format(gpu.id, gpu.load*100, gpu.memoryUtil*100))
        elif attrList is None:
            # if `attrList` was not specified, use the default one
            attrList = [[{'attr':'id','name':'ID'},
                         {'attr':'load','name':'GPU','suffix':'%','transform': lambda x: x*100,'precision':0},
                         {'attr':'memoryUtil','name':'MEM','suffix':'%','transform': lambda x: x*100,'precision':0}],
                        ]
        
    if (not useOldCode):
        if (attrList is not None):
            headerString = ''
            GPUstrings = ['']*len(GPUs)
            for attrGroup in attrList:
                #print(attrGroup)
                for attrDict in attrGroup:
                    headerString = headerString + '| ' + attrDict['name'] + ' '
                    headerWidth = len(attrDict['name'])
                    minWidth = len(attrDict['name'])
                    
                    attrPrecision = '.' + str(attrDict['precision']) if ('precision' in attrDict.keys()) else ''
                    attrSuffix = str(attrDict['suffix']) if ('suffix' in attrDict.keys()) else ''
                    attrTransform = attrDict['transform'] if ('transform' in attrDict.keys()) else lambda x : x
                    for gpu in GPUs:
                        attr = getattr(gpu,attrDict['attr'])
                        
                        attr = attrTransform(attr)
                        
                        if (isinstance(attr,float)):
                            attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
                        elif (isinstance(attr,int)):
                            attrStr = ('{0:d}').format(attr)
                        elif (isinstance(attr,str)):
                            attrStr = attr;
                        elif  (sys.version_info[0] == 2):
                            if (isinstance(attr,unicode)):
                                attrStr = attr.encode('ascii','ignore')
                        else:
                            raise TypeError('Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')
                                            
                        attrStr += attrSuffix
                        
                        minWidth = max(minWidth,len(attrStr))
    
                    headerString += ' '*max(0,minWidth-headerWidth)
                    
                    minWidthStr = str(minWidth - len(attrSuffix))
                    
                    for gpuIdx,gpu in enumerate(GPUs):
                        attr = getattr(gpu,attrDict['attr'])
                        
                        attr = attrTransform(attr)
                        
                        if (isinstance(attr,float)):
                            attrStr = ('{0:'+ minWidthStr + attrPrecision + 'f}').format(attr)
                        elif (isinstance(attr,int)):
                            attrStr = ('{0:' + minWidthStr + 'd}').format(attr)
                        elif (isinstance(attr,str)):
                            attrStr = ('{0:' + minWidthStr + 's}').format(attr);
                        elif (sys.version_info[0] == 2):
                            if (isinstance(attr,unicode)):
                                attrStr = ('{0:' + minWidthStr + 's}').format(attr.encode('ascii','ignore'))
                        else:
                            raise TypeError('Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')
                                            
                        attrStr += attrSuffix
                        
                        GPUstrings[gpuIdx] += '| ' + attrStr + ' '
                                            
                headerString = headerString + '|'
                for gpuIdx,gpu in enumerate(GPUs):
                    GPUstrings[gpuIdx] += '|'
                    
            headerSpacingString = '-' * len(headerString)
            print(headerString)
            print(headerSpacingString)
            for GPUstring in GPUstrings:
                print(GPUstring)


# Generate gpu uuid to id map
gpuUuidToIdMap = {}
try:
    gpus = getGPUs()
    for gpu in gpus:
        gpuUuidToIdMap[gpu.uuid] = gpu.id
    del gpus
except:
    pass

