#ifndef COLLISIONDETINTERNALS_CUH
#define COLLISIONDETINTERNALS_CUH

#include "HashTable.cuh"
#include "Importer.cuh"

struct Voxel {
    // from LSB: first 10 bits are x index, middle 10 bits are y index, last 10 bits are z
    uint32_t index = 0;
    static constexpr float SIZE = 2.0f / (1 << 10);
};

// Bounding box is defined in terms of the starting voxel index, and the number of
// voxels that it includes in each direction
// Then there is a 3 dimensional array of atleast size, which would be used to mark
// if a triangle occupies that particular voxel
struct BoundingBox {
    float start_vertex[3];
    float end_vertex[3];

    uint32_t start_i;
    uint32_t size[3];
    uint32_t capacity[3];
    uint32_t *occupied;

    __device__ void setOccupied(Voxel v);
    __device__ uint32_t getOccupied(int x, int y, int z);
};

__host__ void initHashTable(int numBuckets);
__host__ void initBoundingBox(Mesh mesh);

__device__ Voxel getVoxel(Triangle *t);
__device__ __host__ Voxel getVoxel(float v[3]);
// __device__ void updatePosition(Triangle *t, int mesh_i);
__device__ __host__ void updatePositionVertex(float vertex[3], float trans_mat[4][4]);
__device__ void markCollision(uint32_t voxel_i, uint32_t triangle_i);
void transformAndResetBox();

// __global__ void updateHashTable(Mesh *m, int mesh_i);
// __global__ void updateBoundingBox(Mesh *m, int mesh_i, BoundingBox *box);
__global__ void markCollidingTriangles();

__device__ void updateHashTable(int triangleIndex, int meshIndex, Voxel oldVoxel, Voxel newVoxel);
__device__ void updateBoundingBox(Triangle *t);

#endif /* COLLISIONDETINTERNALS_CUH */