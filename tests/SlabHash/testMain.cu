#include "SlabHash/HashTable.cuh"
#include "errorcheck.h"
#include <cstdio>

__device__ inline int laneID() {
	return threadIdx.x % warpSize;
}

__device__ inline int warpID() {
	return CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x / warpSize);
}

__device__ void read(SlabAlloc * s, ResidentBlock * rb, Address a) {
	uint32_t data1 = *(s->SlabAddress(a, laneID()));
	__syncwarp();
	Address address = __shfl_sync(WARP_MASK, data1, ADDRESS_LANE);
	uint32_t data2 = *(s->SlabAddress(address, laneID()));
	if(laneID() != 31 && (data1 != laneID() || data2 != laneID() + 32))
		printf("Warp %d, Lane %d: Slab 1 - %d, Slab 2 - %d\n", warpID(), laneID(), data1, data2);
}

__device__ float sum_local_rbl_changes = 0.0;

__global__ void kernel(SlabAlloc * s) {
	ResidentBlock rb(s);
	int x = 0;
	int y = 0;
	Address a = rb.warp_allocate(&x), a2 = rb.warp_allocate(&y);
	float avg = ((float)x + (float)y) / 2;
	atomicAdd(&sum_local_rbl_changes, avg);
	if (a == EMPTY_ADDRESS || a2 == EMPTY_ADDRESS) {
		return;
	}
	auto ptr = s->SlabAddress(a, laneID());
	*ptr = laneID();
	if(laneID() == ADDRESS_LANE) {
		*ptr = a2;
	}
	ptr = s->SlabAddress(a2, laneID());
	*ptr = laneID() + 32;
	
	read(s, &rb, a);
}

int main() {
	const ULL numWarps = 1<<13, numSuperBlocks = numWarps >> 8;
	SlabAlloc * s = new SlabAlloc(numSuperBlocks);
	SlabAlloc * d_s;
	gpuErrchk(cudaMalloc(&d_s, sizeof(SlabAlloc)));
	gpuErrchk(cudaMemcpy(d_s, s, sizeof(SlabAlloc), cudaMemcpyDefault));
	int numBlocks = numWarps>>5, threadsPerBlock = 1024;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<28));
	kernel<<<numBlocks,threadsPerBlock>>>(d_s);
	gpuErrchk(cudaDeviceSynchronize());
	float avg_local_rbl_changes = 0.0;
	gpuErrchk(cudaMemcpyFromSymbol(&avg_local_rbl_changes, sum_local_rbl_changes, sizeof(float)));
	avg_local_rbl_changes /= (numWarps << 5);
	printf("Average local_rbl_changes = %f\n", avg_local_rbl_changes);
	gpuErrchk(cudaFree(d_s));
	delete s;
	gpuErrchk(cudaDeviceReset());
}
