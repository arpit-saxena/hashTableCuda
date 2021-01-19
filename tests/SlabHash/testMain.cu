#include "render.cuh"

void rendertest() {
  glm::mat4 init[2] = {
      glm::translate(glm::mat4(1.0f), glm::vec3(0.1f, 0.0f, 0.0f)),
      glm::translate(glm::mat4(1.0f), glm::vec3(-0.1f, 0.0f, 0.0f))};
  glm::mat4 trans[2] = {
      glm::translate(glm::mat4(1.0f), glm::vec3(-0.01f, 0.0f, 0.0f)),
      glm::translate(glm::mat4(1.0f), glm::vec3(0.01f, 0.0f, 0.0f))};

  Mesh mesh = import("models/bunny.ply");
  Mesh h_meshes[2] = {mesh, mesh};

  OpenGLScene scene(h_meshes, init, trans, nullptr);
  scene.render();
}

int main() {
  rendertest();
  gpuErrchk(cudaDeviceReset());
}
