#ifndef COLLISION_MARKER_CUH
#define COLLISION_MARKER_CUH

#include "CollisionDet/Importer.cuh"

namespace collisionMarker {
extern __constant__ Triangle* d_meshes[2];
extern __constant__ int numTriangles[2];
__host__ void init(Mesh meshes[2]);
__device__ void markCollision(uint32_t, uint32_t);
}  // namespace collisionMarker

#endif /* COLLISION_MARKER_CUH */