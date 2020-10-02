#include "render.cuh"

__host__ bool CUDA::detectCollision(Mesh* d_meshes, HashTable* d_h) {
	bool collisionHappened = false;
	return collisionHappened;
}

__device__ void CUDA::updateTrianglePosition(Triangle* currtriangle, int meshIndex, HashTable* d_h, const glm::mat4 transformation_mat) {
	transform(currtriangle, transformation_mat);
}