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


__host__ void CUDA::preprocess() {
	transformAndResetBox();
}

__host__ bool CUDA::detectCollision(Mesh* d_meshes, HashTable* d_h) {
	/* static int frame = 0;
	if (frame++ > 120) {
		int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = 64;
		testmarking<<<numBlocks, threadsPerBlock>>>();
		return true;
	} */ // !< Hope this was just testing and 120 isn't sth important

	dim3 threadsPerBlock = dim3(1, 1, THREADS_PER_BLOCK),
		 numBlocks = dim3(1, 1, 128); //FIXME: Change this >.<
	markCollidingTriangles<<<numBlocks, threadsPerBlock>>>();
	gpuErrchk( cudaDeviceSynchronize() );
	bool collisionHappened = false;
	return collisionHappened;
}

__device__ void CUDA::updateTrianglePosition(Triangle* triangle, int triangleIndex, int meshIndex, HashTable* d_h, const glm::mat4 transformationMat) {
	Voxel oldVoxel, newVoxel;
	if (triangle != nullptr) {
		oldVoxel = getVoxel(triangle);
		transform(triangle, transformationMat);
		newVoxel = getVoxel(triangle);
	}

	updateHashTable(triangleIndex, meshIndex, oldVoxel, newVoxel);
	if (meshIndex == 0) updateBoundingBox(triangle); // FIXME: DIVERGENCE!
}

__constant__ Triangle* collisionMarker::d_meshes[2] = { nullptr, nullptr };
__constant__ int collisionMarker::numTriangles[2];

__host__ void collisionMarker::init(Mesh meshes[2]) {
	const int numMeshes = 2;
	/*int h_numTriangles[numMeshes];
	Triangle* trianglearrs_in_meshes[numMeshes];*/
	for (int i = 0; i < numMeshes; ++i) {
		gpuErrchk(cudaMemcpyToSymbol(numTriangles, &meshes[i].numTriangles, sizeof(int), i * sizeof(int)));
		gpuErrchk(cudaMemcpyToSymbol(d_meshes, &(meshes[i].triangles), sizeof(Triangle *), i*sizeof(Triangle*)));
		/*h_numTriangles[i] = meshes[i].numTriangles;
		trianglearrs_in_meshes[i] = meshes[i].triangles;*/
	}
	/*gpuErrchk(cudaMemcpyToSymbol(numTriangles, h_numTriangles, numMeshes * sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_meshes, trianglearrs_in_meshes, numMeshes * sizeof(Triangle*)));*/
	//printf("In init, meshes={%x, %x}\n", meshes[0].triangles, meshes[1].triangles);
}

__device__ void collisionMarker::markCollision(uint32_t voxel, uint32_t triangle_i)
{
	//printf("Marking voxel %d\n", voxel);
	//printf("in markCollision, d_meshes={%x, %x}\n", d_meshes[0], d_meshes[1]);
	int numT = numTriangles[1];
	if (!(triangle_i < numTriangles[1])) {
		printf("triangle_i=%d, numTriangles={%d,%d}, khela!\n", triangle_i, numTriangles[0], numTriangles[1]);
	}
	assert(triangle_i < numTriangles[1]);
	Triangle* t = d_meshes[1] + triangle_i;
	for (int i = 0; i < 3; ++i) {
		t->vertices[i].hasCollided = 1.0f;
	}
	//printf("Collided: %d\n", triangle_i);
}
