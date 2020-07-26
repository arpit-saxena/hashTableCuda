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

//__device__ void readanddeallocate(Address a) {
//	uint32_t data1 = *(SlabAlloc::SlabAddress(a, laneID()));
//	__syncwarp();
//	Address address = __shfl_sync(WARP_MASK, data1, ADDRESS_LANE);
//	uint32_t data2 = *(SlabAlloc::SlabAddress(address, laneID()));
//	//SlabAlloc::deallocate(a);
//	//SlabAlloc::deallocate(address);
//	if(laneID() != 31 && (data1 != warpID() || data2 != warpID() + (1 << 18)))
//		printf("After writing, Warp %d, Lane %d: Slab 1 - %d, Slab 2 - %d\n", warpID(), laneID(), data1, data2);
//}
//
//__device__ float sum_local_rbl_changes = 0.0;
//__device__ float sum_sqr_local_rbl_changes = 0.0;
//
//__global__ void checkallbitmaps() {
//	//Checking if array SlabAlloc::bitmaps has been copied properly (it most probably has)
//	int i = blockDim.x * blockIdx.x + threadIdx.x;
//	while (i < SlabAlloc::maxSuperBlocks * SuperBlock::numMemoryBlocks) {
//		uint32_t Bitmap = SlabAlloc::bitmaps[i / 32].bitmap[i % 32];
//		if (Bitmap != 0) {
//			printf("SlabAlloc::bitmaps[%d].bitmap[%d] = %x, instead of 0\n", i / 32, i % 32, Bitmap);
//		}
//		i += gridDim.x;
//	}
//}
//
//__global__ void kernel() {
//	ResidentBlock rb;
//	int x = 0;
//	int y = 0;
//	Address a = rb.warp_allocate(&x), a2 = rb.warp_allocate(&y);
//	
//	// Calculation of average local_rbl_changes, and terminating threads for whom any one warp_allocate() fails
//	float avg = ((float)x + (float)y) / 2;
//	atomicAdd(&sum_local_rbl_changes, avg);
//	atomicAdd(&sum_sqr_local_rbl_changes, (avg*avg));
//	if (a == EMPTY_ADDRESS || a2 == EMPTY_ADDRESS) {
//		if (a != EMPTY_ADDRESS)	SlabAlloc::deallocate(a);
//		else if (a2 != EMPTY_ADDRESS)	SlabAlloc::deallocate(a2);
//		return;
//	}
//
//	// Checking if all slabs have been initialized properly
//	uint32_t data1 = *(SlabAlloc::SlabAddress(a, laneID())), data2 = *(SlabAlloc::SlabAddress(a2, laneID()));
//	if((data1 != 0xFFFFFFFF || data2 != 0xFFFFFFFF))
//		printf("Before writing, Warp %d, Lane %d: Slab 1 - %x, Slab 2 - %x\n", warpID(), laneID(), data1, data2);
//
//	auto ptr = SlabAlloc::SlabAddress(a, laneID());
//	*ptr = warpID();
//	if(laneID() == ADDRESS_LANE) {
//		*ptr = a2;
//	}
//	ptr = SlabAlloc::SlabAddress(a2, laneID());
//	*ptr = warpID()+(1<<18);
//	
//	readanddeallocate(a);
//}
//
//void test1() {
//	const ULL log2slabsPerWarp = 0;	// Cannot be greater than SLAB_BITS(10) + MEMORYBLOCK_BITS(8)
//	// Make sure numWarps is big enough so that numSuperBlocks is non-zero
//	const ULL numWarps = 1 << 18, numSuperBlocks = numWarps >> SLAB_BITS + MEMORYBLOCK_BITS - log2slabsPerWarp;
//	SlabAlloc::init();
//	int numBlocks = numWarps>>5, threadsPerBlock = 1024;
//	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<28));
//
//	checkallbitmaps <<< ((SlabAlloc::maxSuperBlocks * SuperBlock::numMemoryBlocks) >> 5), 1024 >>> ();
//	gpuErrchk(cudaDeviceSynchronize());
//	printf("Completed check of array SlabAlloc::bitmaps before running kernel\n");
//
//	kernel<<<numBlocks,threadsPerBlock>>>();
//	gpuErrchk(cudaDeviceSynchronize());
//
//	//checkallbitmaps <<< ((SlabAlloc::maxSuperBlocks * SuperBlock::numMemoryBlocks) >> 5), 1024 >>> ();
//	gpuErrchk(cudaDeviceSynchronize());
//	printf("Completed check of array SlabAlloc::bitmaps after running kernel\n");
//
//	float avg_local_rbl_changes = 0.0, var_local_rbl_changes = 0.0;
//	gpuErrchk(cudaMemcpyFromSymbol(&avg_local_rbl_changes, sum_local_rbl_changes, sizeof(float)));
//	gpuErrchk(cudaMemcpyFromSymbol(&var_local_rbl_changes, sum_sqr_local_rbl_changes, sizeof(float)));
//	avg_local_rbl_changes /= (numWarps << 5);
//	var_local_rbl_changes = var_local_rbl_changes / (numWarps << 5) - (avg_local_rbl_changes * avg_local_rbl_changes);
//	printf("Average local_rbl_changes = %f, Variance in local_rbl_changes=%f\n", avg_local_rbl_changes, var_local_rbl_changes);
//
//	unsigned nsb = 0;
//	gpuErrchk(cudaMemcpyFromSymbol(&nsb, SlabAlloc::numSuperBlocks, sizeof(unsigned int)));
//	printf("Final no. of superblocks: %d\n", nsb);
//	SlabAlloc::destroy();
//}
//
//
//__global__ void kernel2(SlabAlloc * s) {
//	ResidentBlock rb;
//	int x = 0;
//	for(int i = 0; i <= MemoryBlock::numSlabs; ++i) {
//		if(threadIdx.x == 0)	printf("\r%.4d", i);
//		rb.warp_allocate(&x);
//	}
//	SlabAlloc::allocateSuperBlock();
//}
//
//void test2() {
//	const ULL numWarps = 1, numSuperBlocks = 1;
//	SlabAlloc * s = new SlabAlloc(numSuperBlocks);
//	SlabAlloc * d_s;
//	gpuErrchk(cudaMalloc(&d_s, sizeof(SlabAlloc)));
//	gpuErrchk(cudaMemcpy(d_s, s, sizeof(SlabAlloc), cudaMemcpyDefault));
//	int numBlocks = numWarps, threadsPerBlock = 32;
//	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<28));
//
//	kernel2<<<numBlocks,threadsPerBlock>>>(d_s);
//
//	gpuErrchk(cudaFree(d_s));
//}


__managed__ uint32_t search_success = 0;
__managed__ uint32_t delete_success = 0;
__managed__ uint32_t finder_success = 0;

__device__ inline uint32_t Key() {		return blockIdx.x*blockDim.x + threadIdx.x;	}
__device__ inline uint32_t Value() {	return Key()+5;	}

__global__ void kernel3ins(HashTable* h) {
	ResidentBlock rb;
	HashTableOperation op(h, &rb);
	op.run(Instruction::Type::Insert, Key(), Value());
}

/*__global__ void kernel3inscheck(HashTable* h) {
	//ResidentBlock rb;
	uint32_t value = SEARCH_NOT_FOUND;
	HashTableOperation op(h, nullptr);
	op.run(Instruction::Type::Search, Key(), value);
	if (value != SEARCH_NOT_FOUND) {
		atomicAdd(&search_success, 1);
	}
}*/

__device__ void callBack(uint32_t key, uint32_t value) {
	atomicAdd(&finder_success, 1);
}
__device__ void (*d_callBack)(uint32_t, uint32_t) = callBack;
void findvaluescheck(HashTable * h, int numBlocks, int threadsPerBlock) {
	uint32_t * keys = new uint32_t[numBlocks];
	for(int i = 0; i < numBlocks; ++i) {
		keys[i] = i;
	}
	void* h_callBack;
	gpuErrchk(cudaMemcpyFromSymbol(&h_callBack, d_callBack, sizeof(&callBack)));
	h->findvalues(keys, numBlocks, reinterpret_cast<void(*)(uint32_t, uint32_t)>(h_callBack));
	delete[]keys;
}

__global__ void kernel3del(HashTable* h) {
	//ResidentBlock rb;
	HashTableOperation op(h, nullptr);
	op.run(Instruction::Type::Delete, Key(), Value());
}

/*__global__ void kernel3delcheck(HashTable* h) {
	//ResidentBlock rb;
	uint32_t value = SEARCH_NOT_FOUND;
	HashTableOperation op(h, nullptr);
	op.run(Instruction::Type::Search, Key(), value);
	if (value == SEARCH_NOT_FOUND) {
		atomicAdd(&delete_success, 1);
	}
}*/

void test3() {
	const ULL numThreads = 1<<18;
	const ULL numWarps = numThreads >> 5;
	SlabAlloc::init();
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<30));
	
	int no_of_buckets = numThreads / 128;	// avg slabs per bucket ~ 9-10, assuming 1 insert instruction per thread
	HashTable * h = new HashTable(no_of_buckets);
	HashTable * d_h;
	gpuErrchk(cudaMalloc(&d_h, sizeof(HashTable)));
	gpuErrchk(cudaMemcpy(d_h, h, sizeof(HashTable), cudaMemcpyDefault));

	int numBlocks = numWarps>>1, threadsPerBlock = THREADS_PER_BLOCK;
	kernel3ins<<<numBlocks, threadsPerBlock>>>(d_h);
	//kernel3inscheck<<<numBlocks, threadsPerBlock>>>(d_h);
	findvaluescheck(h, numBlocks*threadsPerBlock, threadsPerBlock);
	kernel3del<<<numBlocks, threadsPerBlock>>>(d_h);
	//kernel3delcheck<<<numBlocks, threadsPerBlock>>>(d_h);

	gpuErrchk(cudaFree(d_h));
	delete h;
	SlabAlloc::destroy();

	printf("searcher() success rate = %f%\n", (float)search_success * 100 / (float)numThreads);
	printf("deleter() success rate = %f%\n", (float)delete_success * 100 / (float)numThreads);
	printf("finder() success rate = %f%\n", (float)finder_success*100/(float)numThreads);
}

//
//__global__ void kernel4(SlabAlloc* s) {
//	ResidentBlock rb;
//	Address a = rb.warp_allocate();
//	uint32_t left = threadIdx.x, right = threadIdx.x + 5;
//	uint32_t data[2] = { left, right };
//	if (1 << laneID() & VALID_KEY_MASK) {
//		*(ULL*)(SlabAlloc::SlabAddress(a, laneID())) = *reinterpret_cast<ULL *>(data);
//		//assert(atomicCAS((ULL*)(SlabAlloc::SlabAddress(a, laneID())), (ULL)0xFFFFFFFFFFFFFFFF, *((ULL*)data)) == (ULL)0xFFFFFFFFFFFFFFFF);
//		assert(*(SlabAlloc::SlabAddress(a, laneID())) == left);
//		assert(*(SlabAlloc::SlabAddress(a, laneID()+1)) == right);
//	}
//}
//
//void test4() {
//	const ULL numWarps = 1, numSuperBlocks = 1;
//	SlabAlloc * s = new SlabAlloc(numSuperBlocks);
//	SlabAlloc * d_s;
//	gpuErrchk(cudaMalloc(&d_s, sizeof(SlabAlloc)));
//	gpuErrchk(cudaMemcpy(d_s, s, sizeof(SlabAlloc), cudaMemcpyDefault));
//	int numBlocks = numWarps, threadsPerBlock = 32;
//	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1<<28));
//
//	kernel4<<<numBlocks,threadsPerBlock>>>(d_s);
//
//	gpuErrchk(cudaFree(d_s));
//}

__global__ void somekernel() {
	SlabAlloc::allocateSuperBlock();
}

void unittest() {
	SlabAlloc::init();
	somekernel<<<1<<8, 1024>>>();
	SlabAlloc::destroy();
}

int main() {
	test3();
	gpuErrchk(cudaDeviceReset());
}
