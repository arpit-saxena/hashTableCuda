#ifndef KERNEL_CUH
#define KERNEL_CUH

//#include <vector_types.h>
//#include <vector_functions.hpp>
//#include <cuda_runtime_api.h>
//#include <math_functions.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "SlabHash/HashTable.cuh"
#include <stdio.h>

namespace CUDA{
	extern const unsigned numOfTriangles;
	void runCuda(struct cudaGraphicsResource** vbo_resource, const double currentTime);
	void registerVBO(unsigned int VBO, struct cudaGraphicsResource** vbo_resource);
}

#endif // !KERNEL_CUH
