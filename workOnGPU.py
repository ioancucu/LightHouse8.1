import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

# Create a NumPy array
a = np.array([1, 2, 3], dtype=np.float32)

# Allocate GPU memory
a_gpu = cuda.mem_alloc(a.nbytes)

# Transfer data from CPU to GPU
cuda.memcpy_htod(a_gpu, a)

# Create an empty array for the result
result_gpu = cuda.mem_alloc(a.nbytes)

# Define the CUDA kernel code
kernel_code = """
__global__ void square(float* data_in, float* data_out)
{
    int idx = threadIdx.x;
    data_out[idx] = data_in[idx] * data_in[idx];
}
"""

# Compile and load the CUDA kernel code
module = cuda.module_from_buffer(kernel_code.encode())
kernel = module.get_function("square")

# Launch the kernel
block_size = (len(a), 1, 1)
grid_size = (1, 1, 1)
kernel(a_gpu, result_gpu, block=block_size, grid=grid_size)

# Create an empty array on the CPU for the result
result = np.empty_like(a)

# Transfer the result from GPU to CPU
cuda.memcpy_dtoh(result, result_gpu)

# Print the result
print(result)