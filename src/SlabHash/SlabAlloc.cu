#include "SlabAlloc.cuh"
#include "HashFunction.cuh"
#include <stdio.h>
#include <assert.h>
#include <new>

BlockBitMap::BlockBitMap() {
	memset(bitmap, 0, 32*sizeof(uint32_t));
}

__host__ __device__ Slab::Slab() {
	memset(arr, 0xFF, 32*sizeof(uint32_t));
}

__host__ SlabAlloc::SlabAlloc(int numSuperBlocks = maxSuperBlocks) : initNumSuperBlocks(numSuperBlocks) {
	this -> numSuperBlocks = numSuperBlocks;
	if (numSuperBlocks > maxSuperBlocks) {
		//TODO: Better way to handle this?
		printf("Can't allocate %d super blocks. Max is %d", numSuperBlocks, maxSuperBlocks);
		return;
	}

	cudaMalloc(&superBlocks, maxSuperBlocks*sizeof(SuperBlock *));

	SuperBlock * sb = new SuperBlock();
	for (int i = 0; i < maxSuperBlocks; i++) {
		SuperBlock * temp = nullptr;
		if(i < numSuperBlocks) {
			cudaMalloc(&temp, sizeof(SuperBlock));
			cudaMemcpy(temp, sb , sizeof(SuperBlock), cudaMemcpyDefault);
		}
		cudaMemcpy(superBlocks + i, &temp, sizeof(SuperBlock *), cudaMemcpyDefault);
	}
	delete sb;
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
__device__ int SlabAlloc::allocateSuperBlock() {
	assert(__activemask() == WARP_MASK);
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
				new(newSuperBlock) SuperBlock();
			}
		}
	}

	__syncwarp();
	return __shfl_sync(WARP_MASK, localIdx, workerThreadIdx);
}

__device__ uint32_t * SlabAlloc::SlabAddress(Address addr, uint32_t laneID){
	uint32_t slab_idx = addr & ((1 << SLAB_BITS) - 1);
	uint32_t block_idx = (addr >> SLAB_BITS) & ((1 << MEMORYBLOCK_BITS) - 1);
	uint32_t superBlock_idx = (addr >> (SLAB_BITS + MEMORYBLOCK_BITS));
	return (superBlocks[superBlock_idx]->memoryBlocks[block_idx].slabs[slab_idx].arr) + laneID;
}

__device__ void SlabAlloc::deallocate(Address addr){		//Doesn't need a full warp
	unsigned global_memory_block_no = addr >> SLAB_BITS;
	unsigned memory_unit_no = addr & ((1<<SLAB_BITS)-1);		//addr%1024, basically
	unsigned lane_no = memory_unit_no / 32, slab_no = memory_unit_no % 32;
	int laneID = threadIdx.x % warpSize;
	if(laneID == __ffs(__activemask()) - 1){
		wipeSlab(addr);
		BlockBitMap * resident_bitmap = bitmaps + global_memory_block_no;
		uint32_t * global_bitmap_line = resident_bitmap->bitmap + lane_no;
		atomicAnd(global_bitmap_line, ~(1u << slab_no));
	}
	// TODO Check for divergence here
}

__device__ void SlabAlloc::wipeSlab(Address addr) {			//Doesn't need a full warp
	uint32_t slab_idx = addr & ((1 << SLAB_BITS) - 1);
	uint32_t block_idx = (addr >> SLAB_BITS) & ((1 << MEMORYBLOCK_BITS) - 1);
	uint32_t superBlock_idx = (addr >> (SLAB_BITS + MEMORYBLOCK_BITS));
	auto slabarr = superBlocks[superBlock_idx]->memoryBlocks[block_idx].slabs[slab_idx].arr;
	if (threadIdx.x % warpSize == __ffs(__activemask()) - 1) {
		memset(slabarr, 0xFF, 32 * sizeof(uint32_t));
	}
}

__device__ ResidentBlock::ResidentBlock(SlabAlloc * s) {
	slab_alloc = s;
	resident_changes = -1;
	set();
}

// Needs full warp
__device__ void ResidentBlock::set() {
	if (resident_changes % max_resident_changes == 0 && resident_changes != 0) {
		slab_alloc->allocateSuperBlock();
		#ifndef NDEBUG
		if(threadIdx.x % warpSize == 0)		//DEBUG
			printf(", allocateSuperBlock() called by set(), resident_changes=%d", resident_changes);
		#endif // !NDEBUG
		// resident_changes = -1;	// So it becomes 0 after a memory block is found
	}
	int global_warp_id = CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x/warpSize);
	//unsigned memory_block_no = HashFunction::memoryblock_hash(global_warp_id, resident_changes, SuperBlock::numMemoryBlocks);
	uint32_t total_memory_blocks = slab_alloc->getNumSuperBlocks() * SuperBlock::numMemoryBlocks;
	uint32_t super_memory_block_no = HashFunction::memoryblock_hash(global_warp_id, resident_changes, total_memory_blocks);
	starting_addr = super_memory_block_no << SLAB_BITS;
	++resident_changes;
	int laneID = threadIdx.x % warpSize;
	BlockBitMap * resident_bitmap = slab_alloc->bitmaps + (starting_addr>>SLAB_BITS);
	resident_bitmap_line = resident_bitmap->bitmap[laneID];
}

__device__ Address ResidentBlock::warp_allocate() {
	//TODO remove this loop maybe
	Address allocated_address = EMPTY_ADDRESS;
	const int global_warp_id = CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x / warpSize);
	const int max_allowed_superblock_changes = 2;
	const int max_allowed_memoryblock_changes = max_allowed_superblock_changes * max_resident_changes;
	const int max_local_rbl_changes = max_resident_changes;
	int allocator_thread_no = -1;
	int memoryblock_changes = 0, laneID = threadIdx.x % warpSize;
	for (int local_rbl_changes = 0; local_rbl_changes <= max_local_rbl_changes; ++local_rbl_changes) {		//review the loop termination condition
		int slab_no;
		while (true) {		//Review this loop
			slab_no = HashFunction::unsetbit_index(global_warp_id, local_rbl_changes, resident_bitmap_line);
			allocator_thread_no = HashFunction::unsetbit_index(global_warp_id, local_rbl_changes, ~__ballot_sync(WARP_MASK, slab_no + 1));
			if (allocator_thread_no == -1) { // All memory units are full in the memory block
				if (memoryblock_changes > max_allowed_memoryblock_changes) {
					slab_alloc->status = 1;
					__threadfence();
					int khela = 0;
					assert(khela);
					asm("trap;"); // Kills kernel with error
				}
				set();
				++memoryblock_changes;
			}
			else {
				break;
			}
		}

		if (laneID == allocator_thread_no) {
			uint32_t i = 1 << slab_no;
			auto global_memory_block_no = starting_addr >> SLAB_BITS;
			BlockBitMap* resident_bitmap = slab_alloc->bitmaps + global_memory_block_no;
			uint32_t* global_bitmap_line = resident_bitmap->bitmap + laneID;
			auto oldval = atomicOr(global_bitmap_line, i);
			resident_bitmap_line = oldval | i;
			if ((oldval & i) == 0) {
				allocated_address = starting_addr + (laneID << 5) + slab_no;
			}
		}

		__syncwarp();
		Address toreturn = __shfl_sync(WARP_MASK, allocated_address, allocator_thread_no);
		if (toreturn != EMPTY_ADDRESS) {
			return toreturn;
		}
		// TODO check for divergence on this functions return
	}
	//This means all max_local_rbl_changes attempts to allocate memory failed as the atomicCAS call kept failing
	//Terminate
	slab_alloc->status = 2;
	__threadfence();
	int mahakhela = 0;
	assert(mahakhela);
	asm("trap;");

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
	Address allocated_address = EMPTY_ADDRESS;
	const int global_warp_id = CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x/warpSize);
	const int max_allowed_superblock_changes = 2;
	const int max_allowed_memoryblock_changes = max_allowed_superblock_changes * max_resident_changes;
	const int max_local_rbl_changes = max_resident_changes;
	int allocator_thread_no = -1;
	int memoryblock_changes = 0, laneID = threadIdx.x % warpSize;
	for(/*int local_rbl_changes = 0*/*x = 0; /*local_rbl_changes*/*x <= max_local_rbl_changes; ++(*x) /*++local_rbl_changes*/) {		//review the loop termination condition
		int slab_no;
		auto local_rbl_changes = *x;
		while (true) {		//Review this loop
			slab_no = HashFunction::unsetbit_index(global_warp_id, local_rbl_changes, resident_bitmap_line);
			allocator_thread_no = HashFunction::unsetbit_index(global_warp_id, local_rbl_changes, ~__ballot_sync(WARP_MASK, slab_no + 1));
			if (allocator_thread_no == -1) { // All memory units are full in the memory block
				if (memoryblock_changes > max_allowed_memoryblock_changes) {
					slab_alloc->status = 1;
					__threadfence();
					int khela = 0;
					assert(khela);
					asm("trap;"); // Kills kernel with error
				}
				__syncwarp();
				if (laneID == 0)
					printf("Warp ID=%d, local_rbl_changes=%d, memoryblock_changes=%d, called set()", global_warp_id, *x, memoryblock_changes);
				set();
				if (laneID == 0)
					printf("\n");
				++memoryblock_changes;
			}
			else {
				break;
			}
		}

		if (laneID == allocator_thread_no) {
			uint32_t i = 1 << slab_no;
			auto global_memory_block_no = starting_addr >> SLAB_BITS;
			BlockBitMap* resident_bitmap = slab_alloc->bitmaps + global_memory_block_no;
			uint32_t* global_bitmap_line = resident_bitmap->bitmap + laneID;
			auto oldval = atomicOr(global_bitmap_line, i);
			resident_bitmap_line = oldval | i;
			if ((oldval & i) == 0) {
				allocated_address = starting_addr + (laneID << 5) + slab_no;
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
			return toreturn;
		}
		// TODO check for divergence on this functions return
	}
	//This means all max_local_rbl_changes attempts to allocate memory failed as the atomicCAS call kept failing
	//Terminate
	/*slab_alloc->status = 2;
	__threadfence();
	int mahakhela = 0;
	assert(mahakhela);
	asm("trap;");*/
	__syncwarp();
	if (laneID == allocator_thread_no) {
		printf("warp_allocate() failed for Warp ID=%d. Details of each iteration:\n", global_warp_id);
		for (int i = 0; i < 8; ++i) {
			if (lrc[warp_id_in_block][i] != -1)
				printf("-> Warp ID=%d, local_rbl_changes=%d, oldval=%x, slab_no=%d, allocator_thread_no=%d\n", global_warp_id, lrc[warp_id_in_block][i], ov[warp_id_in_block][i], sn[warp_id_in_block][i], atn[warp_id_in_block][i]);
		}
		printf("-------------------------------------------------------------------------------------------------------\n");
	}
	return EMPTY_ADDRESS; // Will never execute
}
#endif // !NDEBUG
