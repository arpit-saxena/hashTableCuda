#include "SlabAlloc.cuh"

BlockBitMap::BlockBitMap() {
	for(int i = 0; i < 32; ++i){
		bitmap[i] = 0u;
	}
}

SlabAlloc::SlabAlloc(int numSuperBlocks = maxSuperBlocks) {
	this -> numSuperBlocks = numSuperBlocks;
	if (numSuperBlocks > maxSuperBlocks) {
		//TODO: Better way to handle this?
		printf("Can't allocate %d super blocks. Max is %d", numSuperBlocks, maxSuperBlocks);
		return;
	}

	for (int i = 0; i < numSuperBlocks; i++) {
		cudaMalloc(superBlocks + i, sizeof(SuperBlock));
	}
}

__device__ __host__
int SlabAlloc::getNumSuperBlocks() {
	return numSuperBlocks;
}

Address SlabAlloc::makeAddress(uint32_t superBlock_idx, uint32_t memoryBlock_idx, uint32_t slab_idx) {
	return (superBlock_idx << 24)
			+ (memoryBlock_idx << 10)
			+ slab_idx;
}

__device__ int SlabAlloc::allocateSuperBlock() {
	if (numSuperBlocks == maxSuperBlocks) {
		//TODO Better way to handle this?
		printf("Can't allocate more super blocks!");
		status = -1;
		__threadfence();
		asm("trap;");
	}

	// FIXME Race condition!!
	int idx = numSuperBlocks++;
	superBlocks[idx] = (SuperBlock *) malloc(sizeof(SuperBlock));
	return idx;
}

__device__ Slab * SlabAlloc::SlabAddress(Address addr, uint32_t laneID){
	uint32_t slab_idx = addr & ((1 << 10) - 1);
	uint32_t block_idx = (addr >> 10) & ((1 << 14) - 1);
	uint32_t superBlock_idx = (addr >> 24);
	return (superBlocks[superBlock_idx]->memoryBlocks[block_idx].slabs) + slab_idx;
}

__device__ void SlabAlloc::deallocate(Address addr){
	unsigned global_memory_block_no = addr >> 10;
	unsigned memory_unit_no = addr & ((1<<10)-1);		//addr%1024, basically
	unsigned lane_no = memory_unit_no / 32, slab_no = memory_unit_no % 32;
	int laneID = threadIdx.x % warpSize;
	if(laneID == lane_no){
		BlockBitMap * resident_bitmap = bitmaps + global_memory_block_no;
		uint32_t * global_bitmap_line = resident_bitmap->bitmap + lane_no;
		ULL i = ((1llu<<32)-1) ^ (1llu<<slab_no);
		atomicAnd(global_bitmap_line, i);
	}
}

__device__ void ResidentBlock::init(SlabAlloc * s) {
	slab_alloc = s;
	set_superblock();
	set();
}

__device__ void ResidentBlock::set_superblock() {
	int global_warp_id = (blockDim.x * blockIdx.x + threadIdx.x)/warpSize;
	unsigned superblock_no = HashFunction::superblock_hash
			(global_warp_id, resident_changes, slab_alloc->getNumSuperBlocks());
	first_block = superblock_no<<24;
}

__device__ void ResidentBlock::set() {
	//TODO add code for adding superblocks when resident_changes reaches a threshold
	if (resident_changes >= max_resident_changes) {
		first_block = SlabAlloc::makeAddress(slab_alloc->allocateSuperBlock(), 0, 0);
		resident_changes = 0;
	}
	int global_warp_id = (blockDim.x * blockIdx.x + threadIdx.x)/warpSize;
	unsigned memory_block_no = HashFunction::memoryblock_hash(global_warp_id, resident_changes, SuperBlock::numMemoryBlocks);
	starting_addr = first_block + (memory_block_no<<10);
	++resident_changes;
	int laneID = threadIdx.x % warpSize;
	BlockBitMap * resident_bitmap = slab_alloc->bitmaps + (starting_addr>>10);
	resident_bitmap_line = resident_bitmap->bitmap[laneID];
}

__device__ Address ResidentBlock::warp_allocate() {
	//TODO remove this loop maybe
	for(int k = 0; k < 5; ++k) {		//review the loop termination condition
		int allocator_thread_no = 0;
		int x = 0, laneID = threadIdx.x % warpSize;
		int slab_no;
		while(true){		//Review this loop
			uint32_t mask = (1llu << 32) - 1;
			uint32_t flipped_rbl = resident_bitmap_line ^ mask;
			slab_no = __ffs(flipped_rbl);
			allocator_thread_no = __ffs(__ballot_sync(mask, slab_no));
			if(allocator_thread_no == 0){ // All memory units are full in the memory block
				if(x >= 5){
					slab_alloc->status = 1;
					__threadfence();
					asm("trap;"); // Kills kernel with error
				}
				set();
				++x;
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
				return toreturn;
			}
		}
	}
	//This means all 5 attempts to allocate memory failed as the atomicCAS call kept failing
	//Terminate
	slab_alloc->status = 2;
	__threadfence();
	asm("trap;");
}
