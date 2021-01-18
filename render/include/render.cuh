#ifndef RENDER_H
#define RENDER_H

#include <cuda.h>  // To make sure CUDA_VERSION is defined properly
#include <glad/glad.h>
#include <shader_s.h>

#include "SlabHash/HashTable.cuh"
#include "SlabHash/Importer.cuh"
#define GLM_FORCE_CUDA
#include <glm/gtc/type_ptr.hpp>

class OpenGLScene {
 private:
  GLuint VBO[2], VAO[2];
  Shader* sh;
  bool isFirstFrame, collided;
  struct cudaGraphicsResource* vbo_resource[2];
  void registerVBO();
  void prepdraw();
  void runCuda();
  void draw();
  void destroyGLobjs();

 public:
  /*
   * Allocates device memory for the meshes and transfers them from h_meshes
   * Allocates device memory for the initial model matrices as well as the
   * transition model matrices, and transfers the given matrices to device
   * memory Sets the device pointer to the hashtable This does no OpenGL work
   *
   * h_meshes is the array of 2 meshes to be used by the scene(expected to be in
   * host memory)
   *
   * init_model_mat is an array of 2 4*4 matrices that specify the model
   * matrices that are applied on the meshes in the first frame(expected to be
   * in host memory)
   *
   * trans_model_mat is an array of 2 4*4 matrices that specify the model
   * matrices that are applied on the meshes in all frames after the first
   * frame(expected to be in host memory)
   *
   * d_h is a pointer to a HashTable already initialized and stored in device
   * memory
   */
  OpenGLScene(Mesh h_meshes[2], glm::mat4 init_model_mat[2],
              glm::mat4 trans_model_mat[2], HashTable* d_h);

  /*
   * Deallocates all CUDA memory allocated by the constructor
   * This does no OpenGL work
   */
  ~OpenGLScene();

  HashTable* d_h;  // pointer to the HashTable stored in device memory
  Mesh meshes[2];  // 2 Mesh objects with their triangle arrays in device memory
  glm::mat4* init_model_mat;   // 2 model matrices stored in device memory
                               // specifying initial position of the 2 meshes
  glm::mat4* trans_model_mat;  // 2 model matrices stored in device memory
                               // specifying the change in position of the 2
                               // meshes made in every frame
  glm::mat4* identity_model_mat;  // 2 identity model matrices stored in device
                                  // memory, to be used to pause scene animation

  int render();  // Initializes OpenGL, opens a window, and renders the scene by
                 // applying the transformation matrices to the meshes using a
                 // CUDA kernel via CUDA-OpenGL interop then destroys all OpenGL
                 // structures after the window is closed
};

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
 * such as transforming a bounding box. Having a kernel do that is a waste Can
 * also do other pre processing
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

__host__ void launch_kernel(Triangle* buffer[2], unsigned numTriangles[2],
                            Mesh meshes[2], HashTable* d_h,
                            glm::mat4 transformation_mat[2]);

__global__ void triangleKernel(Triangle* buffer0, Triangle* buffer1,
                               unsigned numTriangles0, unsigned numTriangles1,
                               Triangle* mesh0, Triangle* mesh1, HashTable* d_h,
                               glm::mat4* transformation_mat);

__device__ void transform(Triangle* currtriangle,
                          const glm::mat4 transformation_mat);
}  // namespace CUDA

namespace collisionMarker {
extern __constant__ Triangle* d_meshes[2];
extern __constant__ int numTriangles[2];
__host__ void init(Mesh meshes[2]);
__device__ void markCollision(uint32_t, uint32_t);
}  // namespace collisionMarker

#endif  // !RENDER_H
