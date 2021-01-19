#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>  // To make sure CUDA_VERSION is defined properly
#include <glad/glad.h>

#define GLM_FORCE_CUDA
#include <glm/gtc/type_ptr.hpp>

#include "CollisionDet/Importer.cuh"

/*
 * This file is meant to be for some utility functions which are both used by
 * render and collision detection code, since we don't want collision detection
 * code to depend on rendering parts.
 */

namespace CUDA {

__device__ void transform(Triangle* t, const glm::mat4 transformation_mat);

}  // namespace CUDA

#endif /* UTILS_CUH */