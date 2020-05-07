#ifndef SLABALLOC_H
#define SLABALLOC_H

#include <cstdint>

typedef uint32_t Address;	//32 bit address format
typedef unsigned long long ULL;

/*
 * Each memory block has 1024 memory units slabs, so this is essentially an
 * array of bits which each set bit representing that the corresponding slab
 * has been allocated
 */
struct MemoryBlock {
	static const int numMemoryUnits = 1<<10;
	uint32_t bitmap[32];
	MemoryBlock();
};

struct SuperBlock {
	static const int numMemoryBlocks = 1 << 14;
	MemoryBlock memoryBlocks[numMemoryBlocks];
};

// NOTE: Construct the object on host and copy it to the device afterwards to be able
// to run functions
class SlabAlloc {		//A single object of this will reside in global memory
	public:
		static const int maxSuperBlocks = 1 << 8;

	private:
		int numSuperBlocks;
		__device__ SuperBlock * allocateSuperBlock();
		SuperBlock * superBlocks[maxSuperBlocks];

	public:
		uint32_t * beg_address;
		SlabAlloc(int numSuperBlocks);

		__device__ uint32_t * SlabAddress(Address, uint32_t);
		__device__ void deallocate(Address);
};

class ResidentBlock {			//Objects of this will be on thread-local memory
		SlabAlloc * slab_alloc;
		Address starting_addr;		//address of the 1st memory unit of the resident block
		Address first_block;		//address of the 1st memory block of the superblock being used by the warp
		uint32_t resident_bitmap_line;		//local copy of the 32-bit line of the bitmap of the resident block belonging to the lane
		int resident_changes = 0;
	public:
		__device__ void init(SlabAlloc *);
		__device__ void set_superblock();	//Chooses a superblock to be used by the warp
		__device__ void set();		//Chooses a memory block from the current superblock as a resident block
		
		__device__ Address warp_allocate();
};

#endif /* SLABALLOC_H */
