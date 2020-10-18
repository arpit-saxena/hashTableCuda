#include "render.cuh"
#include "SlabHash/CollisionDetInternals.cuh"

__global__ void testmarking() {
	int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	while (global_thread_id < THREADS_PER_BLOCK * 64) {
		collisionMarker::markCollision(0, global_thread_id);

		global_thread_id += gridDim.x * blockDim.x;
	}
}

__host__ void CUDA::initCollisionDet(Mesh meshes[2]) {
	initBoundingBox(meshes[0]);
	initHashTable(2000); //TODO: Base this on number of triangles
}


__host__ void CUDA::preprocess(const glm::mat4 trans_mats[2]) {
	transformAndResetBox(trans_mats[0]);
}

__host__ bool CUDA::detectCollision(Mesh* d_meshes, HashTable* d_h) {
	/* static int frame = 0;
	if (frame++ > 120) {
		int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = 64;
		testmarking<<<numBlocks, threadsPerBlock>>>();
		return true;
	} */ // !< Hope this was just testing and 120 isn't sth important

	int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = 128; //FIXME: Change this >.<
	markCollidingTriangles<<<numBlocks, threadsPerBlock>>>();
	gpuErrchk( cudaDeviceSynchronize() );
	bool collisionHappened = false;
	return collisionHappened;
}

__device__ void CUDA::updateTrianglePosition(Triangle* triangle, int triangleIndex, int meshIndex, HashTable* d_h, const glm::mat4 transformationMat) {
	Voxel oldVoxel = getVoxel(triangle);
	transform(triangle, transformationMat);
	Voxel newVoxel = getVoxel(triangle);

	updateHashTable(triangleIndex, meshIndex, oldVoxel, newVoxel);
	if (meshIndex == 0) updateBoundingBox(triangle); // FIXME: DIVERGENCE!
}

__constant__ Triangle* collisionMarker::d_meshes[2] = { nullptr, nullptr };

__host__ void collisionMarker::init(Mesh meshes[2]) {
	for (int i = 0; i < 2; ++i) {
		gpuErrchk(cudaMemcpyToSymbol(d_meshes, &(meshes[i].triangles), sizeof(Triangle *), i*sizeof(Triangle*)));
	}
}

__device__ void collisionMarker::markCollision(uint32_t voxel, uint32_t triangle_i)
{
	Triangle* t = d_meshes[1] + triangle_i;
	for (int i = 0; i < 3; ++i) {
		t->vertices[i].hasCollided = 1.0f;
	}
}
