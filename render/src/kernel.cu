#include "kernel.cuh"

namespace {
	__constant__ float vertices[] = {
		// vertices           // normal vectors
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
		 0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  0.0f, 1.0f,
		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f, 1.0f,

		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
		-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f
	};
	
	__global__ void triangleKernel(float3* buffer, unsigned numTriangles, const double angle) {
		double sinx, cosx;
		sincospi(angle, &sinx, &cosx);
		int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
		while (global_thread_id < numTriangles) {
			for (int i = 0; i < 3; ++i) {
				unsigned bufferindex = global_thread_id * 6 + i * 2;
				unsigned verticesindex = bufferindex * 3;
				float x = vertices[verticesindex], y = vertices[verticesindex + 1], z = vertices[verticesindex + 2];
				float nx = vertices[verticesindex + 3], ny = vertices[verticesindex + 4], nz = vertices[verticesindex + 5];
				float3 point = make_float3(cosx * x + sinx * z, y, cosx * z - sinx * x);
				float3 normal = make_float3(cosx * nx + sinx * nz, ny, cosx * nz - sinx * nx);
				buffer[bufferindex] = point;
				buffer[bufferindex + 1] = normal;
			}
			global_thread_id += gridDim.x * blockDim.x;
		}
	}
	
	void launch_kernel(float3* buffer, unsigned numTriangles, const double angle) {
		int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = CEILDIV(numTriangles, threadsPerBlock);
		triangleKernel << < numBlocks, threadsPerBlock >> > (buffer, numTriangles, angle);
	}

	double angle = 0; // * pi rads
	double lastTime = -1;
}

const unsigned CUDA::numOfTriangles = 12;
void CUDA::runCuda(struct cudaGraphicsResource** vbo_resource, const double currentTime)
{
	const double speed = 1; // * pi rad/s
	if (lastTime > 0) {
		::angle += speed * (currentTime - ::lastTime);
		while (::angle > 2) {
			::angle -= 2;
		}
	}
	::lastTime = currentTime;
	// map OpenGL buffer object for writing from CUDA
	float3* dptr;
	gpuErrchk(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource));
	::launch_kernel(dptr, num_bytes / (6 * sizeof(float3)), angle);

	// unmap buffer object
	gpuErrchk(cudaGraphicsUnmapResources(1, vbo_resource, 0));

}

void CUDA::registerVBO(unsigned int VBO, struct cudaGraphicsResource** vbo_resource) {
	gpuErrchk(cudaGraphicsGLRegisterBuffer(vbo_resource, VBO, cudaGraphicsRegisterFlagsWriteDiscard));
}