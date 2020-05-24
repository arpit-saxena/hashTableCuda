#include "SlabAlloc.cuh"
#include "HashFunction.cuh"
#include <stdio.h>
#include <assert.h>

BlockBitMap::BlockBitMap() {
	memset(bitmap, 0, 32*sizeof(uint32_t));
}

__host__ __device__ Slab::Slab() {
	memset(arr, (1llu << 32) - 1, 32*sizeof(uint32_t)); // FIXME: Wrong usage of memset
}

SlabAlloc::SlabAlloc(int numSuperBlocks = maxSuperBlocks) {
	this -> numSuperBlocks = numSuperBlocks;
	if (numSuperBlocks > maxSuperBlocks) {
		//TODO: Better way to handle this?
		printf("Can't allocate %d super blocks. Max is %d", numSuperBlocks, maxSuperBlocks);
		return;
	}

	for (int i = 0; i < numSuperBlocks; i++) {
		SuperBlock sb;
		cudaMalloc(superBlocks + i, sizeof(SuperBlock));
		cudaMemcpy(superBlocks + i, &sb , sizeof(SuperBlock), cudaMemcpyDefault);
	}
}

__device__
void SlabAlloc::cleanup() {
	for (int i = 0; i < numSuperBlocks; i++) {
		free(superBlocks[i]);
	}
}

__device__ __host__
int SlabAlloc::getNumSuperBlocks() {
	return numSuperBlocks;
}

__device__
Address SlabAlloc::makeAddress(uint32_t superBlock_idx, uint32_t memoryBlock_idx, uint32_t slab_idx) {
	return (superBlock_idx << 24)
			+ (memoryBlock_idx << 10)
			+ slab_idx;
}

// Currently called with full warp only, so it also assumes full warp
__device__ int SlabAlloc::allocateSuperBlock() {
	assert(__activemask() == 1llu << 32 - 1);
	int workerThreadIdx = 0;
	int localIdx = -1;
	if (threadIdx.x % 32 == workerThreadIdx) {
		int numSuper = numSuperBlocks; // Get a local copy of the variable
		if (numSuper == maxSuperBlocks) {
			localIdx = numSuper - 1; // This is the last super block, deal with it
		} else {
			localIdx = numSuper++;
			SuperBlock * newSuperBlock = (SuperBlock *) malloc(sizeof(SuperBlock));
			SuperBlock * oldSuperBlock = (SuperBlock *) atomicCAS((ULL *) (superBlocks + localIdx), (ULL) nullptr, (ULL) newSuperBlock);
			if (oldSuperBlock != nullptr) {
				free(newSuperBlock);
			} else {
				atomicAdd(&numSuperBlocks, 1);
			}
		}
	}

	__syncwarp();
	return __shfl_sync((1llu << 32) - 1, localIdx, workerThreadIdx);
}

__device__ uint32_t * SlabAlloc::SlabAddress(Address addr, uint32_t laneID){
	uint32_t slab_idx = addr & ((1 << 10) - 1);
	uint32_t block_idx = (addr >> 10) & ((1 << 14) - 1);
	uint32_t superBlock_idx = (addr >> 24);
	return (superBlocks[superBlock_idx]->memoryBlocks[block_idx].slabs[slab_idx].arr) + laneID;
}

__device__ void SlabAlloc::deallocate(Address addr){
	unsigned global_memory_block_no = addr >> 10;
	unsigned memory_unit_no = addr & ((1<<10)-1);		//addr%1024, basically
	unsigned lane_no = memory_unit_no / 32, slab_no = memory_unit_no % 32;
	int laneID = threadIdx.x % warpSize;
	if(laneID == __ffs(__activemask()) - 1){
		BlockBitMap * resident_bitmap = bitmaps + global_memory_block_no;
		uint32_t * global_bitmap_line = resident_bitmap->bitmap + lane_no;
		atomicAnd(global_bitmap_line, ~(1u << slab_no));
	}
	// TODO Check for divergence here
}

__device__ void ResidentBlock::init(SlabAlloc * s) {
	slab_alloc = s;
	resident_changes = -1;
	set_superblock();
	set();
}

__device__ void ResidentBlock::set_superblock() {
	// The line below assume blockDim is divisible by warpSize i.e. 32
	int global_warp_id = (blockDim.x/warpSize) * blockIdx.x + (threadIdx.x/warpSize);
	uint32_t superblock_no = global_warp_id/SuperBlock::numMemoryBlocks;
	first_block = SlabAlloc::makeAddress(superblock_no, 0, 0);
}

// Needs full warp
__device__ void ResidentBlock::set() {
	if (resident_changes >= max_resident_changes) {
		first_block = SlabAlloc::makeAddress(slab_alloc->allocateSuperBlock(), 0, 0);
		resident_changes = -1; // So it becomes 0 after a memory block is found
	}
	int global_warp_id = (blockDim.x/warpSize) * blockIdx.x + (threadIdx.x/warpSize);
	unsigned memory_block_no = HashFunction::memoryblock_hash(global_warp_id, resident_changes, SuperBlock::numMemoryBlocks);
	starting_addr = first_block + (memory_block_no<<10);
	++resident_changes;
	int laneID = threadIdx.x % warpSize;
	BlockBitMap * resident_bitmap = slab_alloc->bitmaps + (starting_addr>>10);
	resident_bitmap_line = resident_bitmap->bitmap[laneID];
}

__device__ Address ResidentBlock::warp_allocate() {
	//TODO remove this loop maybe
	for(int local_rbl_changes = 0; local_rbl_changes < 5; ++local_rbl_changes) {		//review the loop termination condition
		int allocator_thread_no = 0;
		int memoryblock_changes = 0, laneID = threadIdx.x % warpSize;
		int slab_no;
		while(true){		//Review this loop
			uint32_t mask = (1llu << 32) - 1;
			uint32_t flipped_rbl = ~resident_bitmap_line;
			slab_no = __ffs(flipped_rbl);
			allocator_thread_no = __ffs(__ballot_sync(mask, slab_no));
			if(allocator_thread_no == 0){ // All memory units are full in the memory block
				if(memoryblock_changes >= 5){
					slab_alloc->status = 1;
					__threadfence();
					asm("trap;"); // Kills kernel with error
				}
				set();
				++memoryblock_changes;
			} else {
				--allocator_thread_no;		//As it will be a number from 1 to 32, not 0 to 31
				break;
			}
		}

		if(laneID == allocator_thread_no){
			auto new_resident_bitmap_line = resident_bitmap_line ^ (1<<(--slab_no));
			auto global_memory_block_no = starting_addr>>10;
			BlockBitMap * resident_bitmap = slab_alloc->bitmaps + global_memory_block_no;
			uint32_t * global_bitmap_line = resident_bitmap->bitmap + laneID;
			auto oldval = atomicCAS(global_bitmap_line, resident_bitmap_line, new_resident_bitmap_line);
			if(oldval != resident_bitmap_line){
				resident_bitmap_line = oldval;
			}
			else{
				resident_bitmap_line = new_resident_bitmap_line;
				Address toreturn = starting_addr + laneID<<5 + slab_no;
				return toreturn; // FIXME Only one thread returns. All threads should return.
			}
		}

		// TODO check for divergence on this functions return
	}
	//This means all 5 attempts to allocate memory failed as the atomicCAS call kept failing
	//Terminate
	slab_alloc->status = 2;
	__threadfence();
	asm("trap;");

	return 42; // Will never execute
}
