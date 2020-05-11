#include "HashTable.cuh"
#include <bitset>

#define ADDRESS_LANE 31

const uint32_t hashtbl::EMPTY_KEY = (1llu << 32) - 1, 
			hashtbl::EMPTY_VALUE = hashtbl::EMPTY_KEY, 
			hashtbl::SEARCH_NOT_FOUND = hashtbl::EMPTY_KEY, 
			hashtbl::VALID_KEY_MASK = std::bitset<32>(std::string("10101010101010101010101010101000")).to_ulong(), 
			hashtbl::WARP_MASK = hashtbl::EMPTY_KEY;
const Address hashtbl::EMPTY_ADDRESS = hashtbl::EMPTY_KEY;

HashTable::HashTable(int size, SlabAlloc * s) {
	no_of_buckets = size;
	slab_alloc = s;
	cudaMalloc(&base_slabs, no_of_buckets*sizeof(Address));
	int threads_per_block = 32 /* warp size */ , blocks = no_of_buckets;
	hashtbl::init_table<<<blocks, threads_per_block>>>(no_of_buckets, slab_alloc, base_slabs);
}

__global__ void hashtbl::init_table(int no_of_buckets, SlabAlloc * slab_alloc, Address * base_slabs) {
	ResidentBlock rb;	rb.init(slab_alloc);
	int i = blockIdx.x;
	while (i < no_of_buckets) {
		base_slabs[i] = rb.warp_allocate();
		i += gridDim.x;
	}
}

__device__ uint64_t HashTableOperation::makepair(uint32_t key, uint32_t value) {
	return ((uint64_t)key << 32) + value;
}

__device__ uint32_t HashTableOperation::ReadSlab(Address slab_addr, int laneID) {
	return *(SlabAddress(slab_addr, laneID));
}

__device__ uint32_t * HashTableOperation::SlabAddress(Address slab_addr, int laneID) {
	return hashtable->slab_alloc->SlabAddress(slab_addr, laneID);
}

__device__ void HashTableOperation::init(HashTable * h, ResidentBlock * rb, Instruction ins) {
	hashtable = h;
	resident_block = rb;
	instr = ins;
	is_active = true;
	laneID = threadIdx.x % warpSize;
}

__device__ void HashTableOperation::run() {
	uint32_t work_queue = __ballot_sync(hashtbl::WARP_MASK, is_active), old_work_queue = 0;
	while(work_queue != 0) {
		src_lane = __ffs(work_queue);
		assert(src_lane>=1 && src_lane <= 32);
		--src_lane;
		Instruction::Type src_instrtype = static_cast<Instruction::Type>(__shfl_sync(hashtbl::WARP_MASK, instr.type, src_lane));
		src_key = __shfl_sync(hashtbl::WARP_MASK, instr.key, src_lane);
		src_value = __shfl_sync(hashtbl::WARP_MASK, instr.value, src_lane);
		unsigned src_bucket = HashFunction::hash(src_key, hashtable->no_of_buckets);
		if(work_queue != old_work_queue) {
			next = hashtable->base_slabs[src_bucket];
		}
		read_data = ReadSlab(next, laneID);
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
			case Instruction::FindAll:
				finder();
				break;
		}
		old_work_queue = work_queue;
		work_queue = __ballot_sync(hashtbl::WARP_MASK, is_active);
	}
}

__device__ void HashTableOperation::searcher() {
	auto found_lane = __ffs(__ballot_sync(hashtbl::VALID_KEY_MASK, read_data == src_key));
	if(found_lane != 0) {
		--found_lane;
		uint32_t found_value = __shfl_sync(hashtbl::WARP_MASK, read_data, found_lane+1);
		if(laneID == src_lane) {
			instr.value = found_value;
			is_active = false;
		}
	}
	else{
		auto next_ptr = __shfl_sync(hashtbl::WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == hashtbl::EMPTY_ADDRESS) {
			if(laneID == src_lane) {
				instr.value = hashtbl::SEARCH_NOT_FOUND;
				is_active = false;
			}
		}
		else{
			next = next_ptr;
		}
	}
}

__device__ void HashTableOperation::inserter() {
	auto dest_lane = __ffs(__ballot_sync(hashtbl::VALID_KEY_MASK, read_data == hashtbl::EMPTY_KEY));
	if(dest_lane != 0){
		--dest_lane;
		auto empty_pair = makepair(hashtbl::EMPTY_KEY, hashtbl::EMPTY_VALUE);
		if(laneID == src_lane) {
			auto old_pair = atomicCAS((uint64_t *)SlabAddress(next, dest_lane), empty_pair, makepair(src_key, src_value));
			if(old_pair == empty_pair) {
				is_active = false;
			}
		}
	}
	else{
		auto next_ptr = __shfl_sync(hashtbl::WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == hashtbl::EMPTY_ADDRESS) {
			auto new_slab_ptr = resident_block->warp_allocate();
			if(laneID == ADDRESS_LANE) {
				auto oldptr = atomicCAS(SlabAddress(next, laneID), hashtbl::EMPTY_ADDRESS, new_slab_ptr);
				if(oldptr != hashtbl::EMPTY_ADDRESS) {
					hashtable->slab_alloc->deallocate(new_slab_ptr);
				}
			}
		}
		else {
			next = next_ptr;
		}
	}
}

__device__ void HashTableOperation::deleter() {
	uint32_t next_lane_data = __shfl_down_sync(hashtbl::WARP_MASK, read_data, 1);
	auto found_lane = __ffs(__ballot_sync(hashtbl::VALID_KEY_MASK, read_data == src_key && next_lane_data == src_value));
	if(found_lane != 0) {
		--found_lane;
		auto existing_pair = makepair(src_key, src_value);
		if(laneID == src_lane) {
			auto old_pair = atomicCAS((uint64_t *)SlabAddress(next, found_lane)
										, existing_pair, makepair(hashtbl::EMPTY_KEY, hashtbl::EMPTY_VALUE));
			if(old_pair == existing_pair) {
				is_active = false;
			}
		}
	}
	else {
		auto next_ptr = __shfl_sync(hashtbl::WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == hashtbl::EMPTY_ADDRESS) {
			is_active = false;
		}
		else {
			next = next_ptr;
		}
	}
}

// If some warp divergence bullshit crops up, rewrite this function to have 1 lane
// do all the collation of the found values into an array
__device__ void HashTableOperation::finder() {
	Address result_list = resident_block->warp_allocate();
	Address next_result = result_list;
	int no_of_found_values = 0;
	while(next != hashtbl::EMPTY_ADDRESS) {
		read_data = ReadSlab(next, laneID);
		uint32_t found_key_lanes = __ballot_sync(hashtbl::VALID_KEY_MASK, read_data == src_key);
		no_of_found_values += __popc(found_key_lanes);
		uint32_t found_value_lanes = found_key_lanes >> 1;
		uint32_t mask = (hashtbl::WARP_MASK) << (31-laneID);
		uint32_t to_write = (1llu << 32) - 1;
		if(laneID == ADDRESS_LANE) {
			if(read_data != hashtbl::EMPTY_ADDRESS) {
				to_write = resident_block->warp_allocate();
			}
		}
		else if(laneID == ADDRESS_LANE - 1) {
			to_write = __popc(found_key_lanes);
		}
		else if(found_key_lanes & 1 << (31-laneID) ) {
			to_write = __popc(found_value_lanes & mask);
		}
		else if(found_value_lanes & 1 << (31-laneID)) {
			to_write = read_data;
		}
		__syncwarp();
		*SlabAddress(next_result, laneID) = to_write;
		next_result = __shfl_sync(hashtbl::WARP_MASK, to_write, ADDRESS_LANE);
		next = __shfl_sync(hashtbl::WARP_MASK, read_data, ADDRESS_LANE);
	}

	instr.foundvalues = (uint32_t *) malloc(no_of_found_values * sizeof(uint32_t));
	next_result = result_list;
	int no_of_values_added = 0;
	while(next_result != hashtbl::EMPTY_ADDRESS) {
		read_data = ReadSlab(next_result, laneID);
		uint32_t next_lane_data = __shfl_down_sync(hashtbl::WARP_MASK, read_data, 1);
		if(read_data != (1llu << 32) - 1) {
			if(!(laneID & 1) && laneID != ADDRESS_LANE - 1) {
				instr.foundvalues[no_of_values_added + read_data] = next_lane_data;
			}
		}
		__syncwarp();
		no_of_values_added += __shfl_sync(hashtbl::WARP_MASK, read_data, ADDRESS_LANE - 1);
		next_result = __shfl_sync(hashtbl::WARP_MASK, read_data, ADDRESS_LANE);
		hashtable->slab_alloc->deallocate(result_list);
		result_list = next_result;
	}

	is_active = false;
}