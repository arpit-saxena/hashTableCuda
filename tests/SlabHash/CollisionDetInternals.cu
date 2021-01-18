#include <assert.h>

#include "SlabHash/CollisionDetInternals.cuh"
#include "render.cuh"

__device__ BoundingBox *box;
__device__ HashTable *table;

__host__ void initHashTable(int numBuckets) {
  SlabAlloc::init();
  HashTable *h_table = new HashTable(numBuckets);
  HashTable *d_table;
  gpuErrchk(cudaMalloc(&d_table, sizeof(HashTable)));
  gpuErrchk(cudaMemcpy(d_table, h_table, sizeof(HashTable), cudaMemcpyDefault));
  gpuErrchk(cudaMemcpyToSymbol(table, &d_table, sizeof(HashTable *)));
}

// mesh object is in host memory but it's triangles array is stored device
// memory
__host__ void initBoundingBox(Mesh mesh) {
  Triangle *h_triangles =
      (Triangle *)malloc(mesh.numTriangles * sizeof(Triangle));
  gpuErrchk(cudaMemcpy(h_triangles, mesh.triangles,
                       mesh.numTriangles * sizeof(Triangle),
                       cudaMemcpyDefault));

  // If this is a bottleneck, look into GPU based algo for finding minimum
  // Though since this is run only once, CPU based should be fine
  BoundingBox h_box;

  for (int i = 0; i < 3; i++) {
    h_box.start_vertex[i] = h_triangles[0].vertices[0].point[i];
    h_box.end_vertex[i] = h_triangles[0].vertices[0].point[i];
  }

  float centroid[3];
  // printf("Mesh 0 centroids:\n");
  for (int i = 0; i < mesh.numTriangles; i++) {
    for (int j = 0; j < 3; j++) {
      centroid[j] = 0.0f;
      for (int k = 0; k < 3; k++) {
        centroid[j] += h_triangles[i].vertices[k].point[j];
      }
      centroid[j] /= 3.0;
      // printf("%f ", centroid[j]);
    }
    // printf("\n");

    for (int j = 0; j < 3; j++) {
      if (centroid[j] < h_box.start_vertex[j]) {
        h_box.start_vertex[j] = centroid[j];
      }

      if (centroid[j] > h_box.end_vertex[j]) {
        h_box.end_vertex[j] = centroid[j];
      }
    }
  }
  // printf("\n\n");

  // printf("Bounding box:\n");
  // printf("\tStart: %f %f %f\n", h_box.start_vertex[0], h_box.start_vertex[1],
  // h_box.start_vertex[2]); printf("\tEnd: %f %f %f\n\n", h_box.end_vertex[0],
  // h_box.end_vertex[1], h_box.end_vertex[2]);

  h_box.start_i = getVoxel(h_box.start_vertex).index;

  uint32_t mask = ((1 << 10) - 1);
  uint32_t end_i = getVoxel(h_box.end_vertex).index;
  uint32_t capacity[3];
  for (int i = 0; i < 3; i++) {
    uint32_t begin = (h_box.start_i >> 10 * i) & mask;
    uint32_t end = (end_i >> 10 * i) & mask;
    h_box.size[i] = end - begin + 1;
    capacity[i] = h_box.size[i] + 2;
    // ^Only need +1 but I'm scared of floating point errors wrecking stuff up
  }

  uint32_t tmp_size = h_box.size[2];
  h_box.size[2] = CEILDIV(h_box.size[2], 32) * 32;
  capacity[2] = CEILDIV(tmp_size + 2, 32) * 32;

  h_box.totalCapacity = (capacity[0] * capacity[1] * capacity[2]) / 32;

  gpuErrchk(
      cudaMalloc(&h_box.occupied, h_box.totalCapacity * sizeof(uint32_t)));

  BoundingBox *d_box;
  gpuErrchk(cudaMalloc(&d_box, sizeof(BoundingBox)));
  // No need to zero out bounding box since it is done before every draw so the
  // garbage data would get cleared then

  gpuErrchk(cudaMemcpy(d_box, &h_box, sizeof(BoundingBox), cudaMemcpyDefault));
  gpuErrchk(cudaMemcpyToSymbol(box, &d_box, sizeof(BoundingBox *)));
}

__device__ void BoundingBox::setOccupied(Voxel v) {
  // Assuming indices are within bounds
  int mask = (1u << 10) - 1;
  int x = ((v.index >> 0) & mask) - ((start_i >> 0) & mask);
  int y = ((v.index >> 10) & mask) - ((start_i >> 10) & mask);
  int z = ((v.index >> 20) & mask) - ((start_i >> 20) & mask);
  // TODO: Can we just do v.index - start_i?
  assert(((x * size[0] + y) * size[1] + z) / 32 < totalCapacity);
  occupied[((x * size[0] + y) * size[1] + z) / 32] |= 1u << (z % 32);
}

__device__ uint32_t BoundingBox::getOccupied(int x, int y, int z) {
  return occupied[((x * size[0] + y) * size[1] + z) / 32];
}

// Gets a voxel of a triangle.
__device__ Voxel getVoxel(Triangle *t) {
  float centroid[3];
  for (int i = 0; i < 3; i++) {
    centroid[i] = 0.0f;
    for (int j = 0; j < 3; j++) {
      centroid[i] += t->vertices[j].point[i];
    }
    centroid[i] /= 3.0;
  }
  return getVoxel(centroid);
}

__device__ __host__ Voxel getVoxel(float point[3]) {
  Voxel v;
  for (int i = 0; i < 3; i++) {
    assert(point[i] > -1 && point[i] < 1);
    uint32_t index = (int)((point[i] + 1.0) / Voxel::SIZE);
    assert(index <
           (1u << 10));  // Only 10 bits available for index. index >= 0 obv
    v.index |= index << (10 * i);
  }
  return v;
}

__device__ __host__ void updatePositionVertex(float vertex[3],
                                              glm::mat4 *trans_mat) {
  glm::vec4 pt = glm::vec4(vertex[0], vertex[1], vertex[2], 1.0f);
  pt = *trans_mat * pt;
  vertex[0] = pt.x;
  vertex[1] = pt.y;
  vertex[2] = pt.z;
}

__device__ void updateHashTable(int triangleIndex, int meshIndex,
                                Voxel oldVoxel, Voxel newVoxel) {
  ResidentBlock rb;
  HashTableOperation op(table, &rb);
  bool is_active = meshIndex == 1 && oldVoxel.index != newVoxel.index;
  // ^ meshIndex == 1 is since a warp may have triangles from the other mesh too

  op.run(Instruction::Type::Delete, oldVoxel.index, triangleIndex, is_active);
  op.run(Instruction::Type::Insert, newVoxel.index, triangleIndex, is_active);
}

// Assumes bounding box array has already been reset, and position updated
__device__ void updateBoundingBox(Triangle *t) {
  Voxel v = getVoxel(t);
  box->setOccupied(v);
}

__global__ void markCollidingTriangles() {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  for (; x < box->size[0]; x += gridDim.x * blockDim.x) {
    for (; y < box->size[1]; y += gridDim.y * blockDim.y) {
      for (; z < box->size[2]; z += gridDim.z * blockDim.z) {
        uint32_t isOccupied = box->getOccupied(x, y, z);
        ResidentBlock rb;
        HashTableOperation op(table, &rb);

        // For each set bit in isOccupied, we have an associated voxel in which
        // a triangle of mesh 0 resides. For each search voxel, we wish to find
        // all triangles of mesh 1 by searching the hash table

        for (int i = 0; i < 32; i++) {
          assert(__activemask() == WARP_MASK);
          bool is_active = isOccupied & (1 << i);
          if (!is_active)
            continue;  //< No divergence since all lanes have same is_active
          assert(__activemask() == WARP_MASK);
          float voxelMidpoint[3];
          for (int j = 0; j < 3; j++) {
            voxelMidpoint[j] = box->start_vertex[j] + Voxel::SIZE / 2;
          }

          voxelMidpoint[0] += x * Voxel::SIZE;
          voxelMidpoint[1] += y * Voxel::SIZE;
          voxelMidpoint[2] += (z - z % 32 + i) * Voxel::SIZE;

          if (voxelMidpoint[2] >= 1) continue;

          uint32_t voxelIndex = getVoxel(voxelMidpoint).index;

          table->findvalue(voxelIndex, collisionMarker::markCollision);
        }
      }
    }
  }
}

__host__ void transformAndResetBox() {
  glm::mat4 *trans_mat = (glm::mat4 *)malloc(sizeof(glm::mat4));
  // This is potentially a huge time drain since we do this each frame
  gpuErrchk(cudaMemcpy(trans_mat, &CUDA::trans_mats[0], sizeof(glm::mat4),
                       cudaMemcpyDefault));
  BoundingBox h_box;
  BoundingBox *d_box;
  gpuErrchk(cudaMemcpyFromSymbol(&d_box, box, sizeof(BoundingBox *)));
  gpuErrchk(cudaMemcpy(&h_box, d_box, sizeof(BoundingBox), cudaMemcpyDefault));
  updatePositionVertex(h_box.start_vertex, trans_mat);
  updatePositionVertex(h_box.end_vertex, trans_mat);

  // printf("Transformed box:\n");
  // printf("\tStart: %f %f %f\n", h_box.start_vertex[0], h_box.start_vertex[1],
  // h_box.start_vertex[2]); printf("\tEnd: %f %f %f\n\n", h_box.end_vertex[0],
  // h_box.end_vertex[1], h_box.end_vertex[2]);
  h_box.start_i = getVoxel(h_box.start_vertex).index;
  uint32_t end_i = getVoxel(h_box.end_vertex).index;
  uint32_t mask = ((1 << 10) - 1);
  for (int i = 0; i < 3; i++) {
    uint32_t begin = (h_box.start_i >> 10 * i) & mask;
    uint32_t end = (end_i >> 10 * i) & mask;
    h_box.size[i] = end - begin + 1;
  }
  h_box.size[2] = CEILDIV(h_box.size[2], 32) * 32;

  gpuErrchk(
      cudaMemset(h_box.occupied, 0, h_box.totalCapacity * sizeof(uint32_t)));

  gpuErrchk(cudaMemcpy(d_box, &h_box, sizeof(BoundingBox), cudaMemcpyDefault));
  gpuErrchk(cudaMemcpyToSymbol(box, &d_box, sizeof(BoundingBox *)));
}

__host__ void CUDA::checkBox(Mesh mesh) {
  Triangle *h_triangles =
      (Triangle *)malloc(mesh.numTriangles * sizeof(Triangle));
  gpuErrchk(cudaMemcpy(h_triangles, mesh.triangles,
                       mesh.numTriangles * sizeof(Triangle),
                       cudaMemcpyDefault));

  BoundingBox h_box;
  BoundingBox *d_box;
  gpuErrchk(cudaMemcpyFromSymbol(&d_box, box, sizeof(BoundingBox *)));
  gpuErrchk(cudaMemcpy(&h_box, d_box, sizeof(BoundingBox), cudaMemcpyDefault));

  printf("Transformed mesh 0 centroids:\n");
  for (int i = 0; i < mesh.numTriangles; i++) {
    for (int j = 0; j < 3; j++) {
      float centroid_coord = 0.0f;
      for (int k = 0; k < 3; k++) {
        centroid_coord += h_triangles[i].vertices[k].point[j];
      }
      centroid_coord /= 3.0;
      assert(centroid_coord > -1 && centroid_coord < 1);
      printf("%f ", centroid_coord);
      if (!(h_box.start_vertex[j] - centroid_coord <= 1e-7)) {
        printf("\nStart coord: %f", h_box.start_vertex[j]);
        printf("\nDifference: %f", centroid_coord - h_box.start_vertex[j]);
        int mahakhela = 0;
        fflush(stdout);
        assert(mahakhela);
      }
      // assert(h_box.start_vertex[j] <= centroid_coord);
    }
    printf("\n");
  }
}