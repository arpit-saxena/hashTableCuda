#include "CollisionMarker.cuh"
#include "errorcheck.h"

__constant__ Triangle* collisionMarker::d_meshes[2] = {nullptr, nullptr};
__constant__ int collisionMarker::numTriangles[2];

__host__ void collisionMarker::init(Mesh meshes[2]) {
  const int numMeshes = 2;
  /*int h_numTriangles[numMeshes];
  Triangle* trianglearrs_in_meshes[numMeshes];*/
  for (int i = 0; i < numMeshes; ++i) {
    gpuErrchk(cudaMemcpyToSymbol(numTriangles, &meshes[i].numTriangles,
                                 sizeof(int), i * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(d_meshes, &(meshes[i].triangles),
                                 sizeof(Triangle*), i * sizeof(Triangle*)));
    /*h_numTriangles[i] = meshes[i].numTriangles;
    trianglearrs_in_meshes[i] = meshes[i].triangles;*/
  }
  /*gpuErrchk(cudaMemcpyToSymbol(numTriangles, h_numTriangles, numMeshes *
  sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(d_meshes, trianglearrs_in_meshes, numMeshes *
  sizeof(Triangle*)));*/
  // printf("In init, meshes={%x, %x}\n", meshes[0].triangles,
  // meshes[1].triangles);
}

__device__ void collisionMarker::markCollision(uint32_t voxel,
                                               uint32_t triangle_i) {
  // printf("Marking voxel %d\n", voxel);
  // printf("in markCollision, d_meshes={%x, %x}\n", d_meshes[0], d_meshes[1]);
  if (!(triangle_i < numTriangles[1])) {
    printf("triangle_i=%d, numTriangles={%d,%d}, khela!\n", triangle_i,
           numTriangles[0], numTriangles[1]);
  }
  assert(triangle_i < numTriangles[1]);
  Triangle* t = d_meshes[1] + triangle_i;
  for (int i = 0; i < 3; ++i) {
    t->vertices[i].hasCollided = 1.0f;
  }
  // printf("Collided: %d\n", triangle_i);
}
