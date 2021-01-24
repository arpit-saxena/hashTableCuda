#include "CollisionDet/CollisionDetImpl.cuh"
#include "CollisionDet/CollisionDetInterface.cuh"
#include "CollisionDet/CollisionMarker.cuh"
#include "Utils.cuh"

glm::mat4* CUDA::trans_mats = nullptr;
int boundingBoxNumVoxels = 0;

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

  int threadsPerBlock = THREADS_PER_BLOCK,
      numBlocks = /* CEILDIV(boundingBoxNumVoxels * 32, threadsPerBlock) */ 128;
  // printf("%d %d\n", numBlocks, threadsPerBlock);
  markCollidingTriangles<<<numBlocks, threadsPerBlock>>>();
  gpuErrchk(cudaDeviceSynchronize());
  bool collisionHappened = false;
  return collisionHappened;
}

// namespace {
//__global__ void sampleTriangleAndUpdate(const int numPoints, Triangle*
// triangle,
//                                        Triangle oldTriangle,
//                                        const int triangleIndex,
//                                        int meshIndex) {
//  int global_thread_id = __global_warp_id * warpSize + __laneID;
//  const unsigned maxThreadsToRun =
//      CEILDIV(numPoints * numPoints, warpSize) * warpSize;
//  while (global_thread_id < maxThreadsToRun) {
//    Voxel oldVoxel, newVoxel;
//    if (global_thread_id < numPoints * numPoints) {
//      int i = global_thread_id / numPoints, j = global_thread_id % numPoints;
//      if (triangle != nullptr) {
//        auto sampledPoint = CUDA::sampleAPoint(i, j, numPoints, &oldTriangle);
//        oldVoxel = getVoxel(sampledPoint);
//        sampledPoint = CUDA::sampleAPoint(i, j, numPoints, triangle);
//        newVoxel = getVoxel(sampledPoint);
//      }
//    } else {
//      meshIndex = 2;  // So that this thread doesn't update the hashtable
//    }
//    __syncwarp();
//    updateHashTable(triangleIndex, meshIndex, oldVoxel, newVoxel);
//
//    if (meshIndex == 0) updateBoundingBox(newVoxel);  // FIXME: DIVERGENCE!
//    global_thread_id += gridDim.x * blockDim.x;
//  }
//}
//}  // namespace

__device__ void CUDA::updateTrianglePosition(
    Triangle* triangle, int triangleIndex, int meshIndex, HashTable* d_h,
    const glm::mat4 transformationMat) {
  // numPoints^2 points will be sampled from the triangle
  const int numPoints = 1;

  Triangle oldTriangle;
  if (triangle != nullptr) {
    oldTriangle = *triangle;
    transform(triangle, transformationMat);
  }

  /*int threadsPerBlock = THREADS_PER_BLOCK,
      numBlocks = CEILDIV(numPoints * numPoints, threadsPerBlock);
  ::sampleTriangleAndUpdate<<<numBlocks, threadsPerBlock>>>(
      numPoints, triangle, oldTriangle, triangleIndex, meshIndex);*/

  for (int i = 0; i < numPoints; ++i) {
    for (int j = 0; j < numPoints; ++j) {
      Voxel oldVoxel, newVoxel;
      if (triangle != nullptr) {
        auto sampledPoint = sampleAPoint(i, j, numPoints, &oldTriangle);
        oldVoxel = getVoxel(sampledPoint);
        sampledPoint = sampleAPoint(i, j, numPoints, triangle);
        newVoxel = getVoxel(sampledPoint);
      }
      updateHashTable(triangleIndex, meshIndex, oldVoxel, newVoxel);

      if (meshIndex == 0) updateBoundingBox(newVoxel);  // FIXME: DIVERGENCE!
    }
  }
}
