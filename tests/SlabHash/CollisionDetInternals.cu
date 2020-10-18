#include <assert.h>
#include "SlabHash/CollisionDetInternals.cuh"
#include "render.cuh"

#define NUM_BUCKETS 1000

__device__ BoundingBox box;
__device__ HashTable table(1000);

__device__ void BoundingBox::setOccupied(Voxel v) {
	// Assuming indices are within bounds
	int mask = (1u << 10) - 1;
	int x = ((v.index >> 0) & mask) - ((start_i >> 0) & mask);
	int y = ((v.index >> 10) & mask) - ((start_i >> 10) & mask);
	int z = ((v.index >> 20) & mask) - ((start_i >> 20) & mask);
	// TODO: Can we just do v.index - start_i?
	occupied[x][y][z / 32] |= 1u << (z % 32);
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
		uint32_t index = (int) (point[i] / Voxel::SIZE);
		assert(index < (1u << 10)); // Only 10 bits available for index
		v.index |= index << (10 * i);
	}
	return v;
}

// This works for translation matrices. Not sure if it will work for rotation matrices
/* __device__ void updatePosition(Triangle *t, int mesh_i) {
	// Maybe this can be improved, but its a really small matrix
	Triangle t2;
	for (int vertex_i = 0; vertex_i < 3; vertex_i++) {
		for (int i = 0; i < 3; i++) {
			float ans = 0.0f;
			for (int j = 0; j < 3; j++) {
				ans += trans_mat[mesh_i][i][j] * t->vertices[vertex_i][j];
			}
			ans += trans_mat[mesh_i][i][3]; // Since the position vector has 1 in 4th place
			t2.vertices[vertex_i][i] = ans;
		}
	}
	
	memcpy(t, &t2, sizeof(Triangle));
} */

__device__ __host__ void updatePositionVertex(float vertex[3], const glm::mat4 trans_mat) {
	for (int i = 0; i < 3; i++) {
		float ans = 0.0f;
		for (int j = 0; j < 3; j++) {
			ans += trans_mat[i][j] * vertex[j];
		}
		ans += trans_mat[i][3]; // Since the position vector has 1 in 4th place
		vertex[i] = ans;
	}
}

/* __global__ void updateHashTable(Mesh *m, int mesh_i) {
	// FIXME: Assuming enough threads are available
	uint32_t triangle_i = blockDim.x * blockIdx.x + threadIdx.x;
	Triangle *t;
	Voxel oldVoxel, newVoxel;
	if (triangle_i < m->numTriangles) {
		t = &m->triangles[triangle_i];
		oldVoxel = getVoxel(t);
		updatePosition(t, mesh_i);
		newVoxel = getVoxel(t);
	}
	__syncwarp();

	ResidentBlock rb;
	HashTableOperation op(&table, &rb);
	const bool is_active = triangle_i < m->numTriangles && oldVoxel != newVoxel;
	op.run(Instruction::Type::Delete, oldVoxel.index, triangle_i, is_active);
	op.run(Instruction::Type::Insert, newVoxel.index, triangle_i, is_active);
} */

__device__ void updateHashTable(int triangleIndex, int meshIndex, Voxel oldVoxel, Voxel newVoxel) {
    ResidentBlock rb;
    HashTableOperation op(&table, &rb);
    bool is_active = meshIndex == 1 && oldVoxel.index != newVoxel.index;
    // ^ meshIndex == 1 is since a warp may have triangles from the other mesh too
    
    op.run(Instruction::Type::Delete, oldVoxel.index, triangleIndex, is_active);
	op.run(Instruction::Type::Insert, newVoxel.index, triangleIndex, is_active);
}

/* // Assumes bounding box array has already been reset, and position updated
__global__ void updateBoundingBox(Mesh *m, int mesh_i, BoundingBox *box) {
	// FIXME: Assuming enough threads are available
	uint32_t triangle_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (triangle_i < m->numTriangles) {
		Triangle *t = &m->triangles[triangle_i];
		updatePosition(t, mesh_i);
		Voxel v = getVoxel(t);
		box->setOccupied(Voxel v);
	}
} */

// Assumes bounding box array has already been reset, and position updated
__device__ void updateBoundingBox(Triangle *t) {
    Voxel v = getVoxel(t);
    box.setOccupied(v);
}

/* __device__ void markCollision(uint32_t voxel_i, uint32_t triangle_i) {
	//TODO
} */

__global__ void markCollidingTriangles() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	
	for (; x < box.size[0]; x += gridDim.x * blockDim.x) {
        for (; y < box.size[1]; y += gridDim.y * blockDim.y) {
            for (; z < box.size[2] * 32; z += gridDim.z * blockDim.z) {
                uint32_t isOccupied = box.occupied[x][y][z / 32];
                ResidentBlock rb;
                HashTableOperation op(&table, &rb);

                // For each set bit in isOccupied, we have an associated voxel in which 
                // a triangle of mesh 1 resides. For each search voxel, we wish to find
                // all triangles of mesh 0 by searching the hash table

                for (int i = 0; i < 32; i++) {
                    bool is_active = isOccupied & (1 << (32 - i));
                    if (!is_active) continue; //< No divergence since all lanes have same is_active
                    float voxelMidpoint[3];
                    for (int j = 0; j < 3; j++) {
                        voxelMidpoint[j] = box.start_vertex[j] + Voxel::SIZE / 2;
                    }

                    voxelMidpoint[0] += x * Voxel::SIZE;
                    voxelMidpoint[1] += y * Voxel::SIZE;
                    voxelMidpoint[2] += (z - z % 32) * Voxel::SIZE + i;

                    uint32_t voxelIndex = getVoxel(voxelMidpoint).index;

                    table.findvalue(voxelIndex, collisionMarker::markCollision);
                }
            }
        }
    }
}

__host__ void transformAndResetBox(const glm::mat4 trans_mat) {
	BoundingBox h_box;
	gpuErrchk( cudaMemcpy(&h_box, &box, sizeof(BoundingBox), cudaMemcpyDefault) );
	updatePositionVertex(h_box.start_vertex, trans_mat);
	updatePositionVertex(h_box.end_vertex, trans_mat);

	h_box.start_i = getVoxel(h_box.start_vertex).index;
	uint32_t end_i = getVoxel(h_box.end_vertex).index;
	for (int i = 0; i < 3; i++) {
		uint32_t begin = (h_box.start_i >> 10 * i) & ((1u << 10) - 1);
		uint32_t end = (end_i >> 10 * i) & ((1u << 10) - 1);
		h_box.size[i] = end - begin;
	}

	int totalCapacity = h_box.capacity[0] * h_box.capacity[1] * h_box.capacity[2];
	gpuErrchk ( cudaMemset(h_box.occupied, 0, totalCapacity * sizeof(uint32_t)) );

	gpuErrchk( cudaMemcpy(&box, &h_box, sizeof(BoundingBox), cudaMemcpyDefault) );
}