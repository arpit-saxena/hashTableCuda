#include "SlabHash/HashTable.cuh"

__device__ Voxel allvoxelsToFind[TOTAL_NO_OF_THREADS_IN_GRID];

__device__ void markTriangle(uint32_t voxel, uint32_t triangle_key) {
	Triangle::getFromKey(triangle_key).mark();
}

__device__ void nextFrame(Triangle * t, HashTable * h) {
	Voxel oldvoxel = getVoxel(*t);
	updatePosition(t);
	Voxel newvoxel = getVoxel(*t);
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