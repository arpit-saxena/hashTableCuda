#include "SlabHash/HashTable.cuh"
#include "errorcheck.h"
#include <cstdio>
#include <assert.h>

__device__ inline int laneID() {
	return threadIdx.x % warpSize;
}

__device__ inline int warpID() {
	return CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x / warpSize);
}

__device__ void readanddeallocate(SlabAlloc * s, ResidentBlock * rb, Address a) {
	uint32_t data1 = *(s->SlabAddress(a, laneID()));
	__syncwarp();
	Address address = __shfl_sync(WARP_MASK, data1, ADDRESS_LANE);
	uint32_t data2 = *(s->SlabAddress(address, laneID()));
	s->deallocate(a);
	s->deallocate(address);
	if(laneID() != 31 && (data1 != warpID() || data2 != warpID() + (1 << 18)))// + 32))
		printf("After writing, Warp %d, Lane %d: Slab 1 - %d, Slab 2 - %d\n", warpID(), laneID(), data1, data2);
}

__device__ float sum_local_rbl_changes = 0.0;

__global__ void checkallbitmaps(SlabAlloc* s) {
	//Checking if array s->bitmaps has been copied properly (it most probably has)
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < s->maxSuperBlocks * SuperBlock::numMemoryBlocks) {
		uint32_t Bitmap = s->bitmaps[i / 32].bitmap[i % 32];
		if (Bitmap != 0) {
			printf("s->bitmaps[%d].bitmap[%d] = %x, instead of 0\n", i / 32, i % 32, Bitmap);
		}
		i += gridDim.x;
	}
}

__global__ void kernel(SlabAlloc * s) {
	ResidentBlock rb(s);
	int x = 0;
	int y = 0;

	Address a = rb.warp_allocate(&x), a2 = rb.warp_allocate(&y);
	
	// Calculation of average local_rbl_changes, and terminating threads for whom any one warp_allocate() fails
	float avg = ((float)x + (float)y) / 2;
	atomicAdd(&sum_local_rbl_changes, avg);
	if (a == EMPTY_ADDRESS || a2 == EMPTY_ADDRESS) {
		if (a != EMPTY_ADDRESS)	s->deallocate(a);
		else if (a2 != EMPTY_ADDRESS)	s->deallocate(a2);
		return;
	}

	// Checking if all slabs have been initialized properly
	uint32_t data1 = *(s->SlabAddress(a, laneID())), data2 = *(s->SlabAddress(a2, laneID()));
	if((data1 != 0xFFFFFFFF || data2 != 0xFFFFFFFF))
		printf("Before writing, Warp %d, Lane %d: Slab 1 - %x, Slab 2 - %x\n", warpID(), laneID(), data1, data2);

	auto ptr = s->SlabAddress(a, laneID());
	*ptr = warpID();
	if(laneID() == ADDRESS_LANE) {
		*ptr = a2;
	}
	ptr = s->SlabAddress(a2, laneID());
	*ptr = warpID()+(1<<18);// + 32;
	
	readanddeallocate(s, &rb, a);
}

int main() {
	const ULL log2slabsPerWarp = 0;	// Cannot be greater than SLAB_BITS(10) + MEMORYBLOCK_BITS(8)
	// Make sure numWarps is big enough so that numSuperBlocks is non-zero
	const ULL numWarps = 1 << 18, numSuperBlocks = numWarps >> SLAB_BITS + MEMORYBLOCK_BITS - log2slabsPerWarp;
	SlabAlloc * s = new SlabAlloc(numSuperBlocks);
	SlabAlloc * d_s;
	gpuErrchk(cudaMalloc(&d_s, sizeof(SlabAlloc)));
	gpuErrchk(cudaMemcpy(d_s, s, sizeof(SlabAlloc), cudaMemcpyDefault));
	int numBlocks = numWarps>>5, threadsPerBlock = 1024;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<28));

	checkallbitmaps <<< ((s->maxSuperBlocks * SuperBlock::numMemoryBlocks) >> 5), 1024 >>> (d_s);
	gpuErrchk(cudaDeviceSynchronize());
	printf("Completed check of array s->bitmaps before running kernel\n");

	kernel<<<numBlocks,threadsPerBlock>>>(d_s);
	gpuErrchk(cudaDeviceSynchronize());

	checkallbitmaps <<< ((s->maxSuperBlocks * SuperBlock::numMemoryBlocks) >> 5), 1024 >>> (d_s);
	gpuErrchk(cudaDeviceSynchronize());
	printf("Completed check of array s->bitmaps after running kernel\n");

	float avg_local_rbl_changes = 0.0;
	gpuErrchk(cudaMemcpyFromSymbol(&avg_local_rbl_changes, sum_local_rbl_changes, sizeof(float)));
	avg_local_rbl_changes /= (numWarps << 5);
	printf("Average local_rbl_changes = %f\n", avg_local_rbl_changes);
	gpuErrchk(cudaFree(d_s));
	delete s;
	gpuErrchk(cudaDeviceReset());
}
