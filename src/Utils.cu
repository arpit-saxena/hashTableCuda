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

__device__ glm::vec3 CUDA::transform_point(glm::vec3 point,
                                           const glm::mat4 transformation_mat) {
  glm::vec4 t_point = transformation_mat * glm::vec4(point, 1.0f);
  return glm::vec3(t_point);
}

__device__ glm::vec3 CUDA::sampleAPoint(int i, int j, int numPoints,
                                        Triangle* triangle) {
  double r1 = (double)i / (double)numPoints;
  double r2 = (double)j / (double)numPoints;
  glm::dvec3 A = glm::dvec3(triangle->vertices[0].point[0],
                            triangle->vertices[0].point[1],
                            triangle->vertices[0].point[2]),
             B = glm::dvec3(triangle->vertices[1].point[0],
                            triangle->vertices[1].point[1],
                            triangle->vertices[1].point[2]),
             C = glm::dvec3(triangle->vertices[2].point[0],
                            triangle->vertices[2].point[1],
                            triangle->vertices[2].point[2]);
  double sqrtr1 = glm::sqrt(r1);
  glm::dvec3 point =
      (1 - sqrtr1) * A + (sqrtr1 * (1 - r2)) * B + (sqrtr1 * r2) * C;
  return glm::vec3(point);
}
