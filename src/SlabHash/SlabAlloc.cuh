#ifndef SLABALLOC_H
#define SLABALLOC_H

#include <cstdint>
#include "errorcheck.h"

#define WARP_MASK (uint32_t)(0xFFFFFFFF)
#define EMPTY_ADDRESS (Address)WARP_MASK

#define SLAB_BITS 10
#define MEMORYBLOCK_BITS 8
#define SUPERBLOCK_BITS 14

#define CEILDIV(a, b) ((a/b) + (a % b != 0))

#ifdef __CUDACC__
#define __laneID (threadIdx.x % warpSize)
#define __local_warp_id (threadIdx.x / warpSize)
#define __global_warp_id (CEILDIV(blockDim.x, warpSize) * blockIdx.x + __local_warp_id)
#endif // __CUDACC__


/*
 * Address is a representation of 64 bit addresses in 32 bits
 * From the least significant bit
 * - First SLAB_BITS bits represent Slab's index within a memory block
 * - Next MEMORYBLOCK_BITS bits represent Block's index within a super block
 * - Last SUPERBLOCK_BITS bits represent the super block's index
 */
typedef uint32_t Address;	//32 bit address format
typedef unsigned long long ULL;

/*
 * Each memory block has 1024 slabs, so this is essentially an array of bits
 * which each set bit representing that the corresponding slab has been allocated
 */
struct BlockBitMap {
	uint32_t bitmap[32];
	BlockBitMap();
};

struct Slab {
	uint32_t arr[32];
};

struct MemoryBlock {
	static const int numSlabs = 1024;
	Slab slabs[numSlabs];
};

struct SuperBlock {
	static const int numMemoryBlocks = 1 << MEMORYBLOCK_BITS;
	MemoryBlock memoryBlocks[numMemoryBlocks];
};

namespace utilitykernel {
	// Frees up the additional superblocks malloc'ed by allocateSuperBlock() in SlabAlloc
	// Called in SlabAlloc::~SlabAlloc()
	__global__ void clean_superblocks(SuperBlock **, const ULL);
}

// NOTE: Construct the object on host and copy it to the device afterwards to be able
// to run functions. Also, make sure the argument 'numSuperBlocks' passed to the ctor
// must be equal to the total no. of warps in the kernel that will be using this
// object, to ensure no superblocks remain unallocated throughout the lifetime of
// this object
class SlabAlloc {		//A single object of this will reside in global memory
	public:
		static const int maxSuperBlocks = (1 << SUPERBLOCK_BITS) - 1;	// The last superblock will not be allocated, this 
														// ensures EMPTY_ADDRESS is an invalid address
		__device__ static Address makeAddress
			(uint32_t superBlock_idx, uint32_t memoryBlock_idx, uint32_t slab_idx);
	private:
		int numSuperBlocks;
		const int initNumSuperBlocks;
//		SuperBlock ** superBlocks;		// Array of length 'maxSuperBlocks', allocated on the device

	public:
		__host__ SlabAlloc(int numSuperBlocks);
		__host__ ~SlabAlloc();
		BlockBitMap bitmaps[maxSuperBlocks * SuperBlock::numMemoryBlocks];
		__device__ void allocateSuperBlock();
		__device__ __host__ int getNumSuperBlocks();
		__device__ uint32_t * SlabAddress(Address, uint32_t);
		__device__ uint32_t ReadSlab(Address slab_addr, int laneID);
		__device__ void deallocate(Address);
};

namespace allocator {
	extern __constant__ SlabAlloc * slab_alloc;
	extern __constant__ SuperBlock ** superBlocks;
	extern SlabAlloc * h_slab_alloc;
	extern SuperBlock ** h_superBlocks;
	__host__ void init(int numSuperBlocks);
	__host__ void destroy();
}

class ResidentBlock {			//Objects of this will be on thread-local memory
		Address starting_addr;		//address of the 1st memory unit of the resident block

		uint32_t resident_bitmap_line;		//local copy of the 32-bit line of the bitmap of the resident block belonging to the lane
		int resident_changes;

	public:
		static const int max_resident_changes = 1024;

		__device__ ResidentBlock();
		__device__ void set();		//Chooses a new memory block as a resident block
#ifndef NDEBUG
		__device__ Address warp_allocate(int*);
#endif // !NDEBUG
		__device__ Address warp_allocate();
};

#endif /* SLABALLOC_H */
