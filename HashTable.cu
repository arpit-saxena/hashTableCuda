#include "HashTable.cuh"
#include <bitset>

#define ADDRESS_LANE 31

__device__ uint32_t Slab::ReadSlab(Address slab_addr, int laneID) {
	return *(SlabAddress(slab_addr, laneID));
}

HashTable::HashTable(int size, SlabAlloc * s) {
	no_of_buckets = size;
	slab_alloc = s;
	cudaMalloc(&base_slabs, no_of_buckets*sizeof(Address));
	int threads_per_block = 32 /* warp size */ , blocks = no_of_buckets;
	init<<<blocks, threads_per_block>>>();
}

__global__ void HashTable::init() {
	ResidentBlock rb;	rb.init(slab_alloc);
	int i = blockIdx.x;
	while (i < no_of_buckets) {
		base_slabs[i] = rb.warp_allocate();
		i += gridDim.x;
	}
}

__device__ void HashTableOperation::init(HashTable * h, ResidentBlock * rb, Instruction ins) {
	hashtable = h;
	resident_block = rb;
	instr = ins;
	is_active = true;
	std::bitset<32> valid_key_mask(std::string("10101010101010101010101010101000"));
	VALID_KEY_MASK = valid_key_mask.to_ulong();
	WARP_MASK = (1llu << 32) - 1;
}

__device__ void HashTableOperation::run() {
	auto work_queue = __ballot(WARP_MASK, is_active), old_work_queue = 0;
	while(work_queue != 0) {
		src_lane = __ffs(work_queue);
		assert(src_lane>=1 && src_lane <= 32);
		--src_lane;
		Instruction::Type src_instrtype = __shfl_sync(WARP_MASK, instr.type, src_lane);
		src_key = __shfl_sync(WARP_MASK, instr.key, src_lane);
		src_value = __shfl_sync(WARP_MASK, instr.value, src_lane);
		unsigned src_bucket = HashFunction::hash(src_key, hashtable->no_of_buckets);
		if(work_queue != old_work_queue) {
			next = hashtable->base_slabs[src_bucket];
		}
		read_data = Slab::ReadSlab(next, laneID);
		switch(src_instrtype) {
			case Instruction::Insert:
				inserter();
				break;
			case Instruction::Delete:
				deleter();
				break;
			case Instruction::Search:
				searcher();
				break;
		}
		old_work_queue = work_queue;
		work_queue = __ballot(WARP_MASK, is_active);
	}
}
