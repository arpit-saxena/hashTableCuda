#include "HashTable.cuh"
#include <bitset>

#define ADDRESS_LANE 31

__device__ uint32_t Slab::ReadSlab(Address slab_addr, int laneID) {
	return *(SlabAddress(slab_addr, laneID));
}

const uint32_t Slab::EMPTY_KEY = (1llu << 32) - 1, 
			Slab::EMPTY_VALUE = Slab::EMPTY_KEY, 
			Slab::SEARCH_NOT_FOUND = Slab::EMPTY_KEY, 
			Slab::VALID_KEY_MASK = std::bitset<32>(std::string("10101010101010101010101010101000")).to_ulong(), 
			Slab::Slab::WARP_MASK = Slab::EMPTY_KEY;
const Address Slab::EMPTY_ADDRESS = Slab::EMPTY_KEY;

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

__device__ static uint64_t HashTableOperation::makepair(uint32_t key, uint32_t value) {
	return ((uint64_t)key << 32) + value;
}

__device__ void HashTableOperation::init(HashTable * h, ResidentBlock * rb, Instruction ins) {
	hashtable = h;
	resident_block = rb;
	instr = ins;
	is_active = true;
	laneID = threadIdx.x % warpSize;
}

__device__ void HashTableOperation::run() {
	auto work_queue = __ballot(Slab::WARP_MASK, is_active), old_work_queue = 0;
	while(work_queue != 0) {
		src_lane = __ffs(work_queue);
		assert(src_lane>=1 && src_lane <= 32);
		--src_lane;
		Instruction::Type src_instrtype = __shfl_sync(Slab::WARP_MASK, instr.type, src_lane);
		src_key = __shfl_sync(Slab::WARP_MASK, instr.key, src_lane);
		src_value = __shfl_sync(Slab::WARP_MASK, instr.value, src_lane);
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
		work_queue = __ballot(Slab::WARP_MASK, is_active);
	}
}

__device__ void HashTableOperation::searcher() {
	auto found_lane = __ffs(__ballot(Slab::VALID_KEY_MASK, read_data == src_key));
	if(found_lane != 0) {
		--found_lane;
		uint32_t found_value = __shfl_sync(Slab::WARP_MASK, read_data, found_lane+1);
		if(laneID == src_lane) {
			instr.value = found_value;
			is_active = false;
		}
	}
	else{
		auto next_ptr = __shfl_sync(Slab::WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == Slab::EMPTY_ADDRESS) {
			if(laneID == src_lane) {
				instr.value = Slab::SEARCH_NOT_FOUND;
				is_active = false;
			}
		}
		else{
			next = next_ptr;
		}
	}
}

__device__ void HashTableOperation::inserter() {
	auto dest_lane = __ffs(__ballot(Slab::VALID_KEY_MASK, read_data == Slab::EMPTY_KEY));
	if(dest_lane != 0){
		--dest_lane;
		auto empty_pair = makepair(Slab::EMPTY_KEY, Slab::EMPTY_VALUE);
		if(laneID == src_lane) {
			auto old_pair = atomicCAS((uint64_t *)SlabAddress(next, dest_lane), empty_pair, makepair(src_key, src_value));
			if(old_pair == empty_pair) {
				is_active = false;
			}
		}
	}
	else{
		auto next_ptr = __shfl_sync(Slab::WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == Slab::EMPTY_ADDRESS) {
			auto new_slab_ptr = resident_block->warp_allocate();
			if(laneID == ADDRESS_LANE) {
				auto oldptr = atomicCAS(SlabAddress(next, laneID), Slab::EMPTY_ADDRESS, new_slab_ptr);
				if(oldptr != Slab::EMPTY_ADDRESS) {
					resident_block->slab_alloc->deallocate(new_slab_ptr);
				}
			}
		}
		else {
			next = next_ptr;
		}
	}
}

__device__ void HashTableOperation::deleter() {
	uint32_t next_lane_data = __shfl_down_sync(Slab::WARP_MASK, read_data, 1);
	auto found_lane = __ffs(__ballot(Slab::VALID_KEY_MASK, read_data == src_key && next_lane_data == src_value));
	if(found_lane != 0) {
		--found_lane;
		auto existing_pair = makepair(src_key, src_value);
		if(laneID == src_lane) {
			auto old_pair = atomicCAS((uint64_t *)SlabAddress(next, found_lane)
										, existing_pair, makepair(Slab::EMPTY_KEY, Slab::EMPTY_VALUE));
			if(old_pair == existing_pair) {
				is_active = false;
			}
		}
	}
	else {
		auto next_ptr = __shfl_sync(Slab::WARP_MASK, ADDRESS_LANE);
		if(next_ptr == Slab::EMPTY_ADDRESS) {
			is_active = false;
		}
		else {
			next = next_ptr;
		}
	}
}