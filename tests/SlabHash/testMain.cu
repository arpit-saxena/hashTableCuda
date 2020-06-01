#include "SlabHash/HashTable.cuh"
#include "errorcheck.h"
#include <cstdio>

__device__ inline int laneID() {
	return threadIdx.x % warpSize;
}

__device__ void read(SlabAlloc * s, ResidentBlock * rb, Address a) {
	uint32_t data1 = *(s->SlabAddress(a, laneID()));
	__syncwarp();
	Address address = __shfl_sync(WARP_MASK, data1, ADDRESS_LANE);
	uint32_t data2 = *(s->SlabAddress(address, laneID()));
	printf("Warp %d: Slab 1 - %d, Slab 2 - %d\n", laneID(), data1, data2);
}

__global__ void kernel(SlabAlloc * s) {
	ResidentBlock rb(s);
	Address a = rb.warp_allocate(), a2 = rb.warp_allocate();
	auto ptr = s->SlabAddress(a, laneID());
	*ptr = laneID();
	if(laneID() == ADDRESS_LANE) {
		*ptr = a2;
	}
	ptr = s->SlabAddress(a2, laneID());
	*ptr = laneID() + 32;
	
	read(s, &rb, a);

	s->cleanup();
}

int main() {
	const ULL numWarps = 1<<13, numSuperBlocks = numWarps >> 8;
	printf("ASJD\n");
	SlabAlloc * s = new SlabAlloc(numSuperBlocks);
	SlabAlloc * d_s;
	gpuErrchk(cudaMalloc(&d_s, sizeof(SlabAlloc)));
	gpuErrchk(cudaMemcpy(d_s, s, sizeof(SlabAlloc), cudaMemcpyDefault));
	int numBlocks = numWarps>>5, threadsPerBlock = 1024;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<28));
	kernel<<<2,32>>>(d_s);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(d_s));
	delete s;
}
