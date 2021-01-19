#ifndef COLLISION_DET_INTERFACE_CUH
#define COLLISION_DET_INTERFACE_CUH

#include <cuda.h>  // To make sure CUDA_VERSION is defined properly
#include <glad/glad.h>

#define GLM_FORCE_CUDA
#include <glm/gtc/type_ptr.hpp>

#include "CollisionDet/Importer.cuh"
#include "SlabHash/HashTable.cuh"

namespace CUDA {
extern glm::mat4* trans_mats;

/*
 * Called in runCuda() inside the kernel called in launch_kernel(), after the
 * thread updates the triangle's vertices using the transformation
 * matrix(Intended to update the hashtable with the triangle's new voxel)
 * meshIndex can be 0 or 1, indicating which mesh the triangle belongs to
 * d_h is a device pointer pointing to the hashtable stored in device memory
 * NOTE: needs full warp
 */
__device__ void updateTrianglePosition(Triangle* currtriangle,
                                       int triangleIndex, int meshIndex,
                                       HashTable* d_h,
                                       const glm::mat4 transformation_mat);
/*
 * Called in runCuda() after the call to launch_kernel(), to do further
 * processing on all updated triangles Intended to use the updated hashtable to
 * do collision detection d_meshes is a device pointer to the array of 2 meshes
 * that make up the scene, stored in device memory d_h is a device pointer
 * pointing to the hashtable stored in device memory
 */
__host__ bool detectCollision(Mesh* d_meshes, HashTable* d_h);

/*
 * Called in runCuda() before launch_kernel()
 * Allows for associated data structures to be updated whose update is simple
 * such as transforming a bounding box. Having a kernel do that is a waste. It
 * can also do other pre processing
 */
__host__ void preprocess();

/*
 * Called just before entering into the render loop
 * Initialise the data structures associated with collision detection, and any
 * other things which need only be done once
 */
__host__ void initCollisionDet(Mesh meshes[2]);

/*
 * Temp function for debugging
 */
__host__ void checkBox(Mesh mesh);
}  // namespace CUDA

#endif /* COLLISION_DET_INTERFACE_CUH */