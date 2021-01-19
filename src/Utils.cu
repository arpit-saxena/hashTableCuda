#include "Utils.cuh"

__device__ void CUDA::transform(Triangle* t,
                                const glm::mat4 transformation_mat) {
  for (int i = 0; i < 3; ++i) {
    Vertex* v = t->vertices + i;
    glm::vec4 pt = glm::vec4(v->point[0], v->point[1], v->point[2], 1.0f);
    glm::vec4 nm = glm::vec4(v->normal[0], v->normal[1], v->normal[2], 1.0f);
    pt = transformation_mat * pt;
    nm = transformation_mat * nm;
    v->point[0] = pt.x;
    v->normal[0] = nm.x;
    v->point[1] = pt.y;
    v->normal[1] = nm.y;
    v->point[2] = pt.z;
    v->normal[2] = nm.z;
  }
}