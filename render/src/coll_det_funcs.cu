#include "render.cuh"

__global__ void testmarking() {
	int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	while (global_thread_id < THREADS_PER_BLOCK * 64) {
		collisionMarker::markCollision(0, global_thread_id);

		global_thread_id += gridDim.x * blockDim.x;
	}
}

__host__ bool CUDA::detectCollision(Mesh* d_meshes, HashTable* d_h) {
	static int frame = 0;
	if (frame++ > 120) {
		int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = 64;
		testmarking<<<numBlocks, threadsPerBlock>>>();
		return true;
	}
	bool collisionHappened = false;
	return collisionHappened;
}

__device__ void CUDA::updateTrianglePosition(Triangle* currtriangle, int meshIndex, HashTable* d_h, const glm::mat4 transformation_mat) {
	transform(currtriangle, transformation_mat);
}

__constant__ Triangle* collisionMarker::d_meshes[2] = { nullptr, nullptr };

__host__ void collisionMarker::init(Mesh meshes[2]) {
	for (int i = 0; i < 2; ++i) {
		gpuErrchk(cudaMemcpyToSymbol(d_meshes, &(meshes[i].triangles), sizeof(Triangle *), i*sizeof(Triangle*)));
	}
}

__device__ void collisionMarker::markCollision(int voxel, int triangle_i)
{
	Triangle* t = d_meshes[1] + triangle_i;
	for (int i = 0; i < 3; ++i) {
		t->vertices[i].hasCollided = 1.0f;
	}
}
