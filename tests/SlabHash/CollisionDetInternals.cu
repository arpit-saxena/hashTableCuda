#include <assert.h>
#include "SlabHash/CollisionDetInternals.cuh"

// trans_mat[i] is transformation matrix of mesh i
__constant__ float trans_mat[2][4][4];

__device__ void BoundingBox::setOccupied(Voxel v) {
	// Assuming indices are within bounds
	int x = ((v.index >> 0) & ((1u << 10) - 1)) - start_i[0];
	int y = ((v.index >> 10) & ((1u << 10) - 1)) - start_i[1];
	int z = ((v.index >> 20) & ((1u << 10) - 1)) - start_i[2];
	occupied[x][y][z / 32] |= 1u << (z % 32);
}

// Gets a voxel of a triangle.
__device__ Voxel getVoxel(Triangle *t) {
	float centroid[3];
	for (int i = 0; i < 3; i++) {
		centroid[i] = 0.0f;
		for (int j = 0; j < 3; j++) {
			centroid[i] += t->vertices[j][i];
		}
		centroid[i] /= 3.0;
	}
	return getVoxel(centroid);
}

__device__ __host__ Voxel getVoxel(float v[3]) {
	Voxel v;
	for (int i = 0; i < 3; i++) {
		uint32_t index = (int) (v[i] / Voxel::SI);
		assert(index < (1u << 10)); // Only 10 bits available for index
		v.index |= index << (10 * i);
	}
	return v;
}

// This works for translation matrices. Not sure if it will work for rotation matrices
__device__ void updatePosition(Triangle *t, int mesh_i) {
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
}

__device__ __host__ void updatePositionVertex(float vertex[3], float trans_mat[4][4]) {
	for (int i = 0; i < 3; i++) {
		float ans = 0.0f;
		for (int j = 0; j < 3; j++) {
			ans += trans_mat[i][j] * vertex[j];
		}
		ans += trans_mat[i][3]; // Since the position vector has 1 in 4th place
		vertex[i] = ans;
	}
}

__global__ void updateHashTable(Mesh *m, int mesh_i) {
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
}

// Assumes bounding box array has already been reset, and position updated
__global__ void updateBoundingBox(Mesh *m, int mesh_i, BoundingBox *box) {
	// FIXME: Assuming enough threads are available
	uint32_t triangle_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (triangle_i < m->numTriangles) {
		Triangle *t = &m->triangles[triangle_i];
		updatePosition(t, mesh_i);
		Voxel v = getVoxel(t);
		box->setOccupied(Voxel v);
	}
}

__device__ void markCollision(uint32_t voxel_i, uint32_t triangle_i) {
	//TODO
}

__global__ void markCollidingTriangles(BoundingBox *box, HashTable *h) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	
	
}

void transformAndResetBox(float trans_mat[4][4], BoundingBox *d_box) {
	BoundingBox h_box;
	gpuErrchk( cudaMemcpy(&h_box, d_box, sizeof(BoundingBox), cudaMemcpyDefault) );
	updatePositionVertex(h_box.start_vertex, trans_mat);
	updatePositionVertex(h_box.end_vertex, trans_mat);

	h_box.start_i = getVoxel(h_box.start_vertex).index;
	uint32_t end_i = getVoxel(h_box.end_vertex).index;
	for (int i = 0; i < 3; i++) {
		uint32_t beg = (h_box.start_i >> 10 * i) & ((1u << 10) - 1);
		uint32_t end = (end_i >> 10 * i) & ((1u << 10) - 1);
		h_box.size[i] = end - begin;
	}

	int totalCapacity = h_box.capacity[0] * h_box.capacity[1] * h_box.capacity[2];
	gpuErrchk ( cudaMemset(h_box.occupied, 0, totalCapacity * sizeof(uint32_t)) );

	gpuErrchk( cudaMemcpy(d_box, &h_box, sizeof(BoundingBox), cudaMemcpyDefault) );
}

void nextFrame(float h_trans_mat[2][4][4], HashTable *d_h, BoundingBox *d_box, Mesh *d_mesh[2], int numTriangles[2]) {
	transformAndResetBox(h_trans_mat[0], d_box);
	updateBoundingBox<<<CEILDIV(numTriangles[0], THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_mesh[0], 0, d_box);
	updateHashTable<<<CEILDIV(numTriangles[1], THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_mesh[1], 1);

}