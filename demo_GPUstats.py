import GPUstats

# Get all device ids and their processing and memory utiliazion
#(deviceIds, gpuUtil, memUtil) = GPUstats.getGPUs()

# Get the first available GPU
firstGPU = GPUstats.getFirstAvailable()
print('First available GPU id:')
print(firstGPU)

# Get the first available GPU, where processing is less than 1%
firstGPU = GPUstats.getFirstAvailable(maxLoad = 0.01)
print('First available GPU id (load < 1%):')
print(firstGPU)

# Get the first available GPU, where memory usage is less than 1%
firstGPU = GPUstats.getFirstAvailable(maxMemory = 0.01)
print('First available GPU id (memory < 1%):')
print(firstGPU)

# Get the first available GPU, where memory usage is less than 1% and processing is less than 2%
firstGPU = GPUstats.getFirstAvailable(maxMemory = 0.01, maxLoad = 0.02)
print('First available GPU id (memory < 1%, load < 2%):')
print(firstGPU)

# Show what hapens if all GPUs are in use, but setting limits to 0%
firstGPU = GPUstats.getFirstAvailable(maxMemory = 0.0, maxLoad = 0.0)
print('First available GPU id (memory < 0%, load < 0%):')
print(firstGPU)

# Show the utilization of all GPUs in a nice table
GPUstats.showUtilization()

print(GPUstats.getAvailable(order = 'first'))

print(GPUstats.getAvailable(order = 'last'))

print(GPUstats.getAvailable(order = 'random'))

print(GPUstats.getAvailable(order = 'load'))

print(GPUstats.getAvailable(order = 'memory', limit = 2, maxLoad=0))
