#ifndef SLABALLOC_H
#define SLABALLOC_H

#include <cstdint>

/*
 * Address is a representation of 64 bit addresses in 32 bits
 * From the least significant bit
 * - First 10 bits represent Slab's index within a memory block
 * - Next 14 bits represent Block's index within a super block
 * - Last 8 bits represent the super block's index
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
	static const int numMemoryBlocks = 1 << 14;
	MemoryBlock memoryBlocks[numMemoryBlocks];
};

// NOTE: Construct the object on host and copy it to the device afterwards to be able
// to run functions
class SlabAlloc {		//A single object of this will reside in global memory
	public:
		static const int maxSuperBlocks = 1 << 8;
		int status = 0; // Indicates the return code of kernels of allocation code
					   // If a function encounters an error, it sets this to non zero
					   // See https://stackoverflow.com/questions/12521721/crashing-a-kernel-gracefully
		__device__ static Address makeAddress
			(uint32_t superBlock_idx, uint32_t memoryBlock_idx, uint32_t slab_idx);
	private:
		int numSuperBlocks;
		SuperBlock * superBlocks[maxSuperBlocks];

	public:
		SlabAlloc(int numSuperBlocks);
		BlockBitMap bitmaps[maxSuperBlocks * SuperBlock::numMemoryBlocks];
		__device__ int allocateSuperBlock(); // Returns new super block's index
		__device__ __host__ int getNumSuperBlocks();
		__device__ Slab * SlabAddress(Address, uint32_t);
		__device__ void deallocate(Address);
};

class ResidentBlock {			//Objects of this will be on thread-local memory
		SlabAlloc * slab_alloc;
		Address starting_addr;		//address of the 1st memory unit of the resident block
		Address first_block;		//address of the 1st memory block of the superblock being used by the warp
		uint32_t resident_bitmap_line;		//local copy of the 32-bit line of the bitmap of the resident block belonging to the lane
		int resident_changes = 0;

	public:
		static const int max_resident_changes = 6;

		__device__ void init(SlabAlloc *);
		__device__ void set_superblock();	//Chooses a superblock to be used by the warp
		__device__ void set();		//Chooses a memory block from the current superblock as a resident block
		
		__device__ Address warp_allocate();
};

#endif /* SLABALLOC_H */
