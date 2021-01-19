#include "CollisionDet/CollisionDetImpl.cuh"
#include "CollisionDet/CollisionDetInterface.cuh"
#include "CollisionDet/CollisionMarker.cuh"
#include "Utils.cuh"

glm::mat4* CUDA::trans_mats = nullptr;

__global__ void testmarking() {
  int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  while (global_thread_id < THREADS_PER_BLOCK * 64) {
    collisionMarker::markCollision(0, global_thread_id);

    global_thread_id += gridDim.x * blockDim.x;
  }
}

__host__ void CUDA::initCollisionDet(Mesh meshes[2]) {
  initBoundingBox(meshes[0]);
  initHashTable(2000);  // TODO: Base this on number of triangles
}

__host__ void CUDA::preprocess() { transformAndResetBox(); }

__host__ bool CUDA::detectCollision(Mesh* d_meshes, HashTable* d_h) {
  /* static int frame = 0;
  if (frame++ > 120) {
          int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = 64;
          testmarking<<<numBlocks, threadsPerBlock>>>();
          return true;
  } */ // !< Hope this was just testing and 120 isn't sth important

  dim3 threadsPerBlock = dim3(1, 1, THREADS_PER_BLOCK),
       numBlocks = dim3(1, 1, 128);  // FIXME: Change this >.<
  markCollidingTriangles<<<numBlocks, threadsPerBlock>>>();
  gpuErrchk(cudaDeviceSynchronize());
  bool collisionHappened = false;
  return collisionHappened;
}

__device__ void CUDA::updateTrianglePosition(
    Triangle* triangle, int triangleIndex, int meshIndex, HashTable* d_h,
    const glm::mat4 transformationMat) {
  Voxel oldVoxel, newVoxel;
  if (triangle != nullptr) {
    oldVoxel = getVoxel(triangle);
    transform(triangle, transformationMat);
    newVoxel = getVoxel(triangle);
  }

  updateHashTable(triangleIndex, meshIndex, oldVoxel, newVoxel);
  if (meshIndex == 0) updateBoundingBox(triangle);  // FIXME: DIVERGENCE!
}