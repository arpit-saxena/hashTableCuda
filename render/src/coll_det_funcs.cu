#include "render.cuh"

__host__ bool CUDA::detectCollision(Mesh* d_meshes, HashTable* d_h) {
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
	
}
