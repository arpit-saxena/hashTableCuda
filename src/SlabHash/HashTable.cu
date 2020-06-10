#include "HashTable.cuh"
#include "HashFunction.cuh"
#include <assert.h>
#include <stdio.h>

__host__ HashTable::HashTable(int size, SlabAlloc * s) {
	no_of_buckets = size;
	slab_alloc = s;
	cudaMalloc(&base_slabs, no_of_buckets*sizeof(Address));
	int threads_per_block = 32 /* warp size */ , blocks = no_of_buckets;
	utilitykernel::init_table<<<blocks, threads_per_block>>>(no_of_buckets, slab_alloc, base_slabs);
}

__global__ void utilitykernel::init_table(int no_of_buckets, SlabAlloc * slab_alloc, Address * base_slabs) {
	ResidentBlock rb(slab_alloc);
	int warp_id = blockIdx.x;
	while (warp_id < no_of_buckets) {
		base_slabs[warp_id] = rb.warp_allocate();
		warp_id += gridDim.x;
	}
}

__host__ HashTable::~HashTable() {
	cudaFree(base_slabs);
}

__device__ Instruction::~Instruction() {
	if(foundvalues)	free(foundvalues);
	foundvalues = nullptr;
}

__device__ ULL HashTableOperation::makepair(uint32_t key, uint32_t value) {
	uint32_t pair[] = { key, value };
	return *reinterpret_cast<ULL *>(pair);
}

__device__ uint32_t HashTableOperation::ReadSlab(Address slab_addr, int laneID) {
	return *(SlabAddress(slab_addr, laneID));
}

__device__ uint32_t * HashTableOperation::SlabAddress(Address slab_addr, int laneID) {
	return hashtable->slab_alloc->SlabAddress(slab_addr, laneID);
}

__device__ HashTableOperation::HashTableOperation(Instruction * ins, HashTable * h, ResidentBlock * rb, bool is_active) {
	hashtable = h;
	if(rb->slab_alloc != h->slab_alloc) {
		//TODO: Better way to handle this?
		printf("Block:%d,Thread:%d->The resident block and the hashtable passed must have the same SlabAlloc object!\n", blockIdx.x, threadIdx.x);
		rb->slab_alloc->status = 3;
		__threadfence();
		int SlabAllocsnotconsistent = 0;
		assert(SlabAllocsnotconsistent);
		asm("trap;");
	}
	resident_block = rb;
	instr = ins;
	this->is_active = is_active;
	laneID = threadIdx.x % warpSize;
}

__device__ void HashTableOperation::run() {
	uint32_t work_queue = __ballot_sync(WARP_MASK, is_active), old_work_queue = 0;
	while(work_queue != 0) {
		src_lane = __ffs(work_queue);
		assert(src_lane>=1 && src_lane <= 32);
		--src_lane;
		Instruction::Type src_instrtype = static_cast<Instruction::Type>(__shfl_sync(WARP_MASK, instr->type, src_lane));
		src_key = __shfl_sync(WARP_MASK, instr->key, src_lane);
		src_value = __shfl_sync(WARP_MASK, instr->value, src_lane);
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
		work_queue = __ballot_sync(WARP_MASK, is_active);
	}
}

__device__ void HashTableOperation::searcher() {
	auto found_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key));
	if(found_lane != 0) {
		--found_lane;
		uint32_t found_value = __shfl_sync(WARP_MASK, read_data, found_lane+1);
		if(laneID == src_lane) {
			instr->value = found_value;
			is_active = false;
		}
	}
	else{
		auto next_ptr = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == EMPTY_ADDRESS) {
			if(laneID == src_lane) {
				instr->value = SEARCH_NOT_FOUND;
				is_active = false;
			}
		}
		else{
			next = next_ptr;
		}
	}
}

__device__ void HashTableOperation::inserter() {
	auto dest_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == EMPTY_KEY));
	if(dest_lane != 0){
		--dest_lane;
		auto empty_pair = makepair(EMPTY_KEY, EMPTY_VALUE);
		if(laneID == src_lane) {
			auto old_pair = atomicCAS((ULL *)SlabAddress(next, dest_lane), empty_pair, makepair(src_key, src_value));
			if(old_pair == empty_pair) {
				is_active = false;
			}
		}
	}
	else{
		auto next_ptr = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == EMPTY_ADDRESS) {
			auto new_slab_ptr = resident_block->warp_allocate();
			if(laneID == ADDRESS_LANE) {
				auto oldptr = atomicCAS(SlabAddress(next, laneID), EMPTY_ADDRESS, new_slab_ptr);
				if(oldptr != EMPTY_ADDRESS) {
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
	uint32_t next_lane_data = __shfl_down_sync(WARP_MASK, read_data, 1);
	auto found_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key && next_lane_data == src_value));
	if(found_lane != 0) {
		--found_lane;
		auto existing_pair = makepair(src_key, src_value);
		if(laneID == src_lane) {
			auto old_pair = atomicCAS((ULL *)SlabAddress(next, found_lane)
										, existing_pair, makepair(EMPTY_KEY, EMPTY_VALUE));
			if(old_pair == existing_pair) {
				is_active = false;
			}
		}
	}
	else {
		auto next_ptr = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == EMPTY_ADDRESS) {
			if(laneID == src_lane) {
				is_active = false;
			}
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
	while(next != EMPTY_ADDRESS) {
		read_data = ReadSlab(next, laneID);
		uint32_t found_key_lanes = __ballot_sync(VALID_KEY_MASK, read_data == src_key);
		no_of_found_values += __popc(found_key_lanes);
		uint32_t found_value_lanes = found_key_lanes << 1;
		uint32_t mask = (WARP_MASK) >> (31-laneID);
		uint32_t to_write = 0xFFFFFFFF;
		next = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		uint32_t nextslab = EMPTY_ADDRESS;
		if(next != EMPTY_ADDRESS) {
			nextslab = resident_block->warp_allocate();
		}
		if (laneID == ADDRESS_LANE) {
			to_write = nextslab;
		}
		else if(laneID == ADDRESS_LANE - 1) {
			to_write = __popc(found_key_lanes);
		}
		else if(found_key_lanes & 1 << laneID ) {
			to_write = __popc(found_value_lanes & mask);
		}
		else if(found_value_lanes & 1 << laneID) {
			to_write = read_data;
		}
		__syncwarp();
		*SlabAddress(next_result, laneID) = to_write;
		next_result = nextslab;
	}

	if(laneID == src_lane) {
		if(no_of_found_values != 0) {
			instr->foundvalues = (uint32_t *) malloc(no_of_found_values * sizeof(uint32_t));
			if(instr->foundvalues == nullptr) {
				instr->findererror = 1;
			}
			instr->no_of_found_values = no_of_found_values;
		}
		is_active = false;
	}
	__syncwarp();
	uint32_t * result_arr = (uint32_t *)__shfl_sync(WARP_MASK, (ULL)instr->foundvalues, src_lane);
	next_result = result_list;
	int no_of_values_added = 0;
	while(next_result != EMPTY_ADDRESS) {
		read_data = ReadSlab(next_result, laneID);
		uint32_t next_lane_data = __shfl_down_sync(WARP_MASK, read_data, 1);
		if(read_data != 0xFFFFFFFF) {
			if(1 << laneID & VALID_KEY_MASK && result_arr != nullptr) {
				assert(no_of_values_added + read_data < no_of_found_values);
				result_arr[no_of_values_added + read_data] = next_lane_data;
			}
		}
		__syncwarp();
		no_of_values_added += __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE - 1);
		next_result = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		hashtable->slab_alloc->deallocate(result_list);
		result_list = next_result;
	}
}
