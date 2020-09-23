#include <assert.h>
#include "SlabHash/HashTable.cuh"
#include "SlabHash/Importer.h"

// trans_mat[i] is transformation matrix of mesh i
__constant__ float trans_mat[2][4][4];
Mesh mesh[2];

// Gets a voxel of a triangle. Assumes pointer t is non-null
__device__ Voxel getVoxel(Triangle *t) {
	Voxel v;
	for (int i = 0; i < 3; i++) {
		float coord = 0.0f;
		for (int j = 0; j < 3; j++) {
			coord += t->vertices[j][i];
		}
		coord /= 3.0;
		v.indices[i] = (int) (coord / Voxel::SIZE);
	}
	return v;
}

// This works for translation matrices. Not sure if it will work for rotation matrices
void updatePosition(Triangle *t, int mesh_i) {
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

__device__ void BoundingBox::setOccupied(Voxel v) {
	for (int i = 0; i < 3; i++) {
		assert(v.indices[i] >= start_i[i] && v.indices[i] < start_i[i] + size[i]);
	}

	int x = v.indices[0] - start_i[0];
	int y = v.indices[1] - start_i[1];
	int z = v.indices[2] - start_i[2];
	occupied[x][y][z] = true;
}

__device__ Voxel allvoxelsToFind[TOTAL_NO_OF_THREADS_IN_GRID];

__device__ void markTriangle(uint32_t voxel, uint32_t triangle_key) {
	Triangle::getFromKey(triangle_key).mark();
}

__device__ void nextFrame(Triangle * t, int mesh_i, HashTable * h) {
	Voxel oldVoxel = getVoxel(*t);
	updatePosition(t, mesh_i);
	Voxel newVoxel = getVoxel(*t);

	if (mesh_i == 0) { // Using bounding box of first mesh initially
		mesh[0].box.setOccupied(newVoxel);
	}
	__syncthreads();

	// For bounding box approach, the hash table needs to only be updated for triangles
	// of the second mesh.
	// TODO below

	ResidentBlock rb;
	HashTableOperation op(h, &rb);
	// Could make the below 2 to run in parallel
	op.run(Instruction::Type::Delete, oldvoxel, t->getKey(), oldvoxel != newvoxel);
	op.run(Instruction::Type::Insert, newvoxel, t->getKey(), oldvoxel != newvoxel);
	__shared__ unsigned numVoxelsToFind = 0;
	__shared__ Voxel voxelsToFind[THREADS_PER_BLOCK];
	voxelsToFind[threadIdx.x] = EMPTY_KEY;
	if(oldvoxel != newvoxel) {
		voxelsToFind[atomicAdd_block(&numVoxelsToFind, 1)] = newvoxel;	// atomicAdd_block only available on devices with compute capability 6.x
	}
	__syncthreads();
	allvoxelsToFind[blockIdx.x*blockDim.x+threadIdx.x] = voxelsToFind[threadIdx.x];
	__syncthreads();
	if(threadIdx.x == 0) {
		h->findvalues(allvoxelsToFind + blockIdx.x*blockDim.x, numVoxelsToFind, markTriangle);
	}
}