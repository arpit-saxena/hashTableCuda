#include "SlabAlloc.cuh"

BitMap::BitMap() {
	for(int i = 0; i < 32; ++i){
		bitmap[i] = 0;
	}
}

SlabAlloc::SlabAlloc(int ns=256) {
	Ns = ns;
	ULL superblock_size = (1<<7)*Nm*Nu;	//1 memory unit = 2^7 bytes, 1024 memory units = 1 memory block, 2^14 memory blocks = 1 superblock
	cudaMalloc(&beg_address, Ns*superblock_size);

	unsigned long no_of_bitmaps = Ns*Nm;
	BitMap * temp = new BitMap[no_of_bitmaps];
	cudaMalloc(&bitmaps, no_of_bitmaps*sizeof(BitMap));
	cudaMemcpy(bitmaps, temp, no_of_bitmaps*sizeof(BitMap), cudaMemcpyHostToDevice);
	delete []temp;
}

__device__ uint32_t * SlabAlloc::SlabAddress(Address addr, int laneID){
	return beg_address + addr<<5 + laneID;
}

__device__ void SlabAlloc::deallocate(Address addr){
	unsigned global_memory_block_no = addr >> 10;
	unsigned memory_unit_no = addr & ((1<<10)-1);		//addr%1024, basically
	unsigned lane_no = memory_unit_no / 32, slab_no = memory_unit_no % 32;
	int laneID = threadIdx.x % warpSize;
	if(laneID == lane_no){
		BitMap * resident_bitmap = bitmaps + global_memory_block_no;
		uint32_t * global_bitmap_line = resident_bitmap->bitmap + lane_no;
		ULL i = ((1<<32)-1) ^ (1<<slab_no);
		auto oldval = atomicAnd(global_bitmap_line, i);
		//How do I update the resident bitmap line now? Or do I have to? Also, what if oldval's slab_no'th bit was already zero?
	}
}

__device__ void ResidentBlock::init(SlabAlloc * s) {
	slab_alloc = s;
	set_superblock();
	set();
}

__device__ void ResidentBlock::set_superblock() {
	int global_warp_id = (blockDim.x * blockIdx.x + threadIdx.x)/warpSize;
	unsigned superblock_no = HashFunction::superblock_hash(global_warp_id, resident_changes, slab_alloc->Ns);
	first_block = superblock_no<<24;
}

__device__ void ResidentBlock::set() {
	//TODO add code for adding superblocks when resident_changes reaches a threshold
	int global_warp_id = (blockDim.x * blockIdx.x + threadIdx.x)/warpSize;
	unsigned memory_block_no = HashFunction::memoryblock_hash(global_warp_id, resident_changes, SlabAlloc::Nm);
	starting_addr = first_block + (memory_block_no<<10);
	++resident_changes;
	int laneID = threadIdx.x % warpSize;
	BitMap * resident_bitmap = slab_alloc->bitmaps + (starting_addr>>10);
	resident_bitmap_line = resident_bitmap->bitmap[laneID];
}

__device__ Address ResidentBlock::warp_allocate() {
	for(int k = 0; k < 5; ++k) {		//review the loop termination condition
		int allocator_thread_no = 0;
		int x = 0, laneID = threadIdx.x % warpSize;
		while(true){		//Review this loop
			ULL i = (1<<32) - 1;
			uint32_t flipped_rbl = resident_bitmap_line ^ (uint32_t)i;
			int slab_no = __ffs(flipped_rbl);
			allocator_thread_no = __ffs(__ballot_sync(i, slab_no));
			if(allocator_thread_no == 0){
				if(x >= 5){
					//Terminate program
					std::exit(1);
				}
				set();
				++x;
			}
			else{
				--allocator_thread_no;		//As it will be a number from 1 to 32, not 0 to 31
				break;
			}
		}
		if(laneID == allocator_thread_no){
			new_resident_bitmap_line = resident_bitmap_line ^ (1<<(--slab_no));
			unsigned global_memory_block_no = starting_addr>>10;
			BitMap * resident_bitmap = slab_alloc->bitmaps + global_memory_block_no;
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
	std::exit(1);
}