#include "SlabAlloc.cuh"
#include "HashFunction.cuh"
#include "errorcheck.h"
#include <stdio.h>
#include <assert.h>
#include <new>

BlockBitMap::BlockBitMap() {
	memset(bitmap, 0, 32*sizeof(uint32_t));
}

__host__ SlabAlloc::SlabAlloc(int numSuperBlocks = maxSuperBlocks) : initNumSuperBlocks(numSuperBlocks) {
	this -> numSuperBlocks = numSuperBlocks;
	if (numSuperBlocks > maxSuperBlocks) {
		//TODO: Better way to handle this?
		printf("Can't allocate %d super blocks. Max is %d", numSuperBlocks, maxSuperBlocks);
		return;
	}

	cudaMalloc(&superBlocks, maxSuperBlocks*sizeof(SuperBlock *));

	for (int i = 0; i < maxSuperBlocks; i++) {
		SuperBlock * temp = nullptr;
		if(i < numSuperBlocks) {
			cudaMalloc(&temp, sizeof(SuperBlock));
		}
		cudaMemcpy(superBlocks + i, &temp, sizeof(SuperBlock *), cudaMemcpyDefault);
	}
}

__host__ SlabAlloc::~SlabAlloc() {
	int size = maxSuperBlocks - initNumSuperBlocks;
	if (size != 0) {
		int threadsPerBlock = 64, numBlocks = CEILDIV(size, threadsPerBlock);
		utilitykernel::clean_superblocks<<<numBlocks, threadsPerBlock>>>(superBlocks + initNumSuperBlocks, size);
	}

	SuperBlock **  h_superBlocks = new SuperBlock *[initNumSuperBlocks];
	cudaMemcpy(h_superBlocks, superBlocks, initNumSuperBlocks*sizeof(SuperBlock *), cudaMemcpyDefault);
	for (int i = 0; i < initNumSuperBlocks; i++) {
		if(h_superBlocks[i])	cudaFree(h_superBlocks[i]);
	}
	delete h_superBlocks;
	cudaFree(superBlocks);
}

__global__
void utilitykernel::clean_superblocks(SuperBlock ** superBlocks, const ULL size) {
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	while(threadID < size) {
		if(superBlocks[threadID])	free(superBlocks[threadID]);
		superBlocks[threadID] = nullptr;
		threadID += gridDim.x * blockDim.x;
	}
}

__constant__ SlabAlloc * allocator::slab_alloc = nullptr;
SlabAlloc * allocator::h_slab_alloc = nullptr;

__host__ void allocator::init(int numSuperBlocks) {
	h_slab_alloc = new SlabAlloc(numSuperBlocks);
	SlabAlloc * d_s;
	gpuErrchk(cudaMalloc(&d_s, sizeof(SlabAlloc)));
	gpuErrchk(cudaMemcpy(d_s, h_slab_alloc, sizeof(SlabAlloc), cudaMemcpyDefault));
	gpuErrchk(cudaMemcpyToSymbol(slab_alloc, &d_s, sizeof(SlabAlloc *)));
}

__host__ void allocator::destroy() {
	gpuErrchk(cudaFree(slab_alloc));
	delete h_slab_alloc;
}

__device__ __host__
int SlabAlloc::getNumSuperBlocks() {
	return numSuperBlocks;
}

__device__
Address SlabAlloc::makeAddress(uint32_t superBlock_idx, uint32_t memoryBlock_idx, uint32_t slab_idx) {
	return (superBlock_idx << (SLAB_BITS + MEMORYBLOCK_BITS))
			+ (memoryBlock_idx << SLAB_BITS)
			+ slab_idx;
}

// Currently called with full warp only, so it also assumes full warp
__device__ void SlabAlloc::allocateSuperBlock() {
	assert(__activemask() == WARP_MASK);
	if (__laneID == 0) {
		if (numSuperBlocks < maxSuperBlocks) {
			SuperBlock * newSuperBlock = (SuperBlock *) malloc(sizeof(SuperBlock));
			if (newSuperBlock == nullptr) {
				/*this->status = 3;
				int OutOfMemory = 0;
				printf("Finally, %d superblocks\n", numSuperBlocks);
				assert(OutOfMemory);
				asm("trap;");*/
				return;
			}
			SuperBlock * oldSuperBlock = (SuperBlock *) atomicCAS((ULL *) (superBlocks + numSuperBlocks), (ULL) nullptr, (ULL) newSuperBlock);
			if (oldSuperBlock != nullptr) {
				free(newSuperBlock);
			} else {
				atomicAdd(&numSuperBlocks, 1);
			}
		}
	}
}

__device__ uint32_t * SlabAlloc::SlabAddress(Address addr, uint32_t laneID){
	uint32_t slab_idx = addr & ((1 << SLAB_BITS) - 1);
	uint32_t block_idx = (addr >> SLAB_BITS) & ((1 << MEMORYBLOCK_BITS) - 1);
	uint32_t superBlock_idx = (addr >> (SLAB_BITS + MEMORYBLOCK_BITS));
	return (superBlocks[superBlock_idx]->memoryBlocks[block_idx].slabs[slab_idx].arr) + laneID;
}

__device__ uint32_t SlabAlloc::ReadSlab(Address slab_addr, int laneID) {
	return *(SlabAddress(slab_addr, laneID));
}

__device__ void SlabAlloc::deallocate(Address addr){		//Doesn't need a full warp
	if(__laneID == __ffs(__activemask()) - 1){
		unsigned global_memory_block_no = addr >> SLAB_BITS;
		unsigned memory_unit_no = addr & ((1<<SLAB_BITS)-1);		//addr%1024, basically
		unsigned lane_no = memory_unit_no / 32, slab_no = memory_unit_no % 32;
		BlockBitMap * resident_bitmap = bitmaps + global_memory_block_no;
		uint32_t * global_bitmap_line = resident_bitmap->bitmap + lane_no;
		atomicAnd(global_bitmap_line, ~(1u << slab_no));
	}
	// TODO Check for divergence here
}

__device__ ResidentBlock::ResidentBlock(){
	resident_changes = -1;
	set();
}

// Needs full warp
__device__ void ResidentBlock::set() {
	if (resident_changes % max_resident_changes == 0 && resident_changes != 0) {
		allocator::slab_alloc->allocateSuperBlock();
		#ifndef NDEBUG
		if(__laneID == 0)		//DEBUG
			printf("\tset()->allocateSuperBlock() called by set(), resident_changes=%d\n", resident_changes);
		#endif // !NDEBUG
		// resident_changes = -1;	// So it becomes 0 after a memory block is found
	}
	//unsigned memory_block_no = HashFunction::memoryblock_hash(__global_warp_id, resident_changes, SuperBlock::numMemoryBlocks);
	uint32_t super_memory_block_no = HashFunction::memoryblock_hash(__global_warp_id, resident_changes,
					allocator::slab_alloc->getNumSuperBlocks() * SuperBlock::numMemoryBlocks/*total_memory_blocks*/);
#ifndef NDEBUG
	//if (__laneID == 0 && resident_changes != -1)		//DEBUG
//		printf("\tset()->super_memory_block_no=hash(__global_warp_id=%d, resident_changes=%d, total_memory_blocks=%d)=%d\n", __global_warp_id, resident_changes, slab_alloc->getNumSuperBlocks() * SuperBlock::numMemoryBlocks, super_memory_block_no);
#endif // !NDEBUG

	starting_addr = super_memory_block_no << SLAB_BITS;
	++resident_changes;
	BlockBitMap * resident_bitmap = allocator::slab_alloc->bitmaps + super_memory_block_no;
	resident_bitmap_line = resident_bitmap->bitmap[__laneID];
}

__device__ Address ResidentBlock::warp_allocate() {
	//TODO remove this loop maybe
	const int max_local_rbl_changes = max_resident_changes;
	int memoryblock_changes = 0;
	for (int local_rbl_changes = 0; local_rbl_changes <= max_local_rbl_changes; ++local_rbl_changes) {		//review the loop termination condition
		int slab_no, allocator_thread_no;
		while (true) {		//Review this loop
			slab_no = HashFunction::unsetbit_index(__global_warp_id, local_rbl_changes, resident_bitmap_line);
			allocator_thread_no = HashFunction::unsetbit_index(__global_warp_id, local_rbl_changes, ~__ballot_sync(WARP_MASK, slab_no + 1));
			if (allocator_thread_no == -1) { // All memory units are full in the memory block
				const int max_allowed_memoryblock_changes = 2 /*max_allowed_superblock_changes*/ * max_resident_changes;
				if (memoryblock_changes > max_allowed_memoryblock_changes) {
					int khela = 0;
					assert(khela);
				}
				set();
				++memoryblock_changes;
			}
			else {
				break;
			}
		}

		Address allocated_address = EMPTY_ADDRESS;
		if (__laneID == allocator_thread_no) {
			uint32_t i = 1 << slab_no;
			auto global_memory_block_no = starting_addr >> SLAB_BITS;
			BlockBitMap* resident_bitmap = allocator::slab_alloc->bitmaps + global_memory_block_no;
			uint32_t* global_bitmap_line = resident_bitmap->bitmap + __laneID;
			auto oldval = atomicOr(global_bitmap_line, i);
			resident_bitmap_line = oldval | i;
			if ((oldval & i) == 0) {
				allocated_address = starting_addr + (__laneID << 5) + slab_no;
			}
		}

		__syncwarp();
		Address toreturn = __shfl_sync(WARP_MASK, allocated_address, allocator_thread_no);
		if (toreturn != EMPTY_ADDRESS) {
			*(allocator::slab_alloc->SlabAddress(toreturn, __laneID)) = EMPTY_ADDRESS;
			return toreturn;
		}
		// TODO check for divergence on this functions return
	}
	//This means all max_local_rbl_changes attempts to allocate memory failed as the atomicCAS call kept failing
	//Terminate
	int mahakhela = 0;
	assert(mahakhela);

	return EMPTY_ADDRESS; // Will never execute
}

#ifndef NDEBUG
__device__ Address ResidentBlock::warp_allocate(int * x) {		//DEBUG
	__shared__ int lrc[32][8];
	__shared__ int sn[32][8];
	__shared__ int atn[32][8];
	__shared__ uint32_t ov[32][8];
	__syncwarp();
	int warp_id_in_block = threadIdx.x / warpSize;
	for (int i = 0; i < 8; ++i)
		lrc[warp_id_in_block][i] = -1;
	//TODO remove this loop maybe
	const int max_local_rbl_changes = max_resident_changes;
	int memoryblock_changes = 0;
	for(/*int local_rbl_changes = 0*/*x = 0; /*local_rbl_changes*/*x <= max_local_rbl_changes; ++(*x) /*++local_rbl_changes*/) {		//review the loop termination condition
		int slab_no, allocator_thread_no;
		auto local_rbl_changes = *x;
		while (true) {		//Review this loop
			slab_no = HashFunction::unsetbit_index(__global_warp_id, local_rbl_changes, resident_bitmap_line);
			allocator_thread_no = HashFunction::unsetbit_index(__global_warp_id, local_rbl_changes, ~__ballot_sync(WARP_MASK, slab_no + 1));
			if (allocator_thread_no == -1) { // All memory units are full in the memory block
				const int max_allowed_memoryblock_changes = 2 /*max_allowed_superblock_changes*/ * max_resident_changes;
				if (memoryblock_changes > max_allowed_memoryblock_changes) {
					int khela = 0;
					assert(khela);
				}
				__syncwarp();
				//if (__laneID == 0)
				//printf("Warp ID=%d, local_rbl_changes=%d, memoryblock_changes=%d, called set()\n", __global_warp_id, *x, memoryblock_changes);
				set();
				++memoryblock_changes;
			}
			else {
				break;
			}
		}

		Address allocated_address = EMPTY_ADDRESS;
		if (__laneID == allocator_thread_no) {
			uint32_t i = 1 << slab_no;
			auto global_memory_block_no = starting_addr >> SLAB_BITS;
			BlockBitMap* resident_bitmap = allocator::slab_alloc->bitmaps + global_memory_block_no;
			uint32_t* global_bitmap_line = resident_bitmap->bitmap + __laneID;
			auto oldval = atomicOr(global_bitmap_line, i);
			resident_bitmap_line = oldval | i;
			if ((oldval & i) == 0) {
				allocated_address = starting_addr + (__laneID << 5) + slab_no;
			}
			else {
				lrc[warp_id_in_block][*x] = *x;
				sn[warp_id_in_block][*x] = slab_no;
				atn[warp_id_in_block][*x] = allocator_thread_no;
				ov[warp_id_in_block][*x] = oldval;
			}
		}

		__syncwarp();
		Address toreturn = __shfl_sync(WARP_MASK, allocated_address, allocator_thread_no);
		if (toreturn != EMPTY_ADDRESS) {
			//uint32_t* ptr = allocator::slab_alloc->SlabAddress(toreturn, __laneID);
			*(allocator::slab_alloc->SlabAddress(toreturn, __laneID)) = EMPTY_ADDRESS;
			return toreturn;
		}
		// TODO check for divergence on this functions return
	}
	//This means all max_local_rbl_changes attempts to allocate memory failed as the atomicCAS call kept failing
	//Terminate
	/*allocator::slab_alloc->status = 2;
	__threadfence();
	int mahakhela = 0;
	assert(mahakhela);
	asm("trap;");*/
	__syncwarp();
	if (__laneID == 0) {
		printf("warp_allocate() failed for Warp ID=%d. Details of each iteration:\n", __global_warp_id);
		for (int i = 0; i < 8; ++i) {
			if (lrc[warp_id_in_block][i] != -1)
				printf("-> Warp ID=%d, local_rbl_changes=%d, oldval=%x, slab_no=%d, allocator_thread_no=%d\n", __global_warp_id, lrc[warp_id_in_block][i], ov[warp_id_in_block][i], sn[warp_id_in_block][i], atn[warp_id_in_block][i]);
		}
		printf("-------------------------------------------------------------------------------------------------------\n");
	}
	return EMPTY_ADDRESS; // Will never execute
}
#endif // !NDEBUG
