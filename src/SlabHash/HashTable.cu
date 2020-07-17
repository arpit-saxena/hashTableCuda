#include "HashTable.cuh"
#include "HashFunction.cuh"
#include <assert.h>
#include <stdio.h>

__host__ HashTable::HashTable(int size, SlabAlloc * s) {
	no_of_buckets = size;
	slab_alloc = s;
	cudaMalloc(&base_slabs, no_of_buckets*sizeof(Address));
	int threads_per_block = 64 /* warp size*2 */ , blocks = no_of_buckets/(threads_per_block/32);
	utilitykernel::init_table<<<blocks, threads_per_block>>>(no_of_buckets, slab_alloc, base_slabs);
}

__global__ void utilitykernel::init_table(int no_of_buckets, SlabAlloc * slab_alloc, Address * base_slabs) {
	ResidentBlock rb(slab_alloc);
	int warp_id = CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x / warpSize);;
	while (warp_id < no_of_buckets) {
		base_slabs[warp_id] = rb.warp_allocate();
		warp_id += gridDim.x;
	}
}

__host__ HashTable::~HashTable() {
	cudaFree(base_slabs);
}

__device__ ULL HashTableOperation::makepair(uint32_t key, uint32_t value) {
	uint32_t pair[] = { key, value };
	return *reinterpret_cast<ULL *>(pair);
}

__device__ uint32_t HashTableOperation::ReadSlab(Address slab_addr, int laneID) {
	return hashtable->slab_alloc->ReadSlab(slab_addr, laneID);
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
}

__device__ void HashTableOperation::run() {
	uint32_t work_queue = __ballot_sync(WARP_MASK, is_active), old_work_queue = 0;
	Instruction::Type src_instrtype;
	while(work_queue != 0) {
		if(work_queue != old_work_queue) {
			src_lane = __ffs(work_queue);
			assert(src_lane>=1 && src_lane <= 32);
			--src_lane;
			src_instrtype = static_cast<Instruction::Type>(__shfl_sync(WARP_MASK, instr->type, src_lane));
			src_key = __shfl_sync(WARP_MASK, instr->key, src_lane);
			src_value = __shfl_sync(WARP_MASK, instr->value, src_lane);
			unsigned src_bucket = HashFunction::hash(src_key, hashtable->no_of_buckets);
			next = hashtable->base_slabs[src_bucket];
		}
		read_data = ReadSlab(next, __laneID);
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
		work_queue = __ballot_sync(WARP_MASK, is_active);
	}
}

__device__ void HashTableOperation::searcher() {
	auto found_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key));
	if(found_lane != 0) {
		--found_lane;
		uint32_t found_value = __shfl_sync(WARP_MASK, read_data, found_lane+1);
		if(__laneID == src_lane) {
			instr->value = found_value;
			is_active = false;
		}
	}
	else{
		auto next_ptr = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		if(next_ptr == EMPTY_ADDRESS) {
			if(__laneID == src_lane) {
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
		if(__laneID == src_lane) {
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
			if(__laneID == ADDRESS_LANE) {
				auto oldptr = atomicCAS(SlabAddress(next, __laneID), EMPTY_ADDRESS, new_slab_ptr);
				if(oldptr != EMPTY_ADDRESS) {
					hashtable->slab_alloc->deallocate(new_slab_ptr);
					new_slab_ptr = oldptr;
				}
			}
			__syncwarp();
			next = __shfl_sync(WARP_MASK, new_slab_ptr, ADDRESS_LANE);
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
		if(__laneID == src_lane) {
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
			if(__laneID == src_lane) {
				is_active = false;
			}
		}
		else {
			next = next_ptr;
		}
	}
}

__host__ void HashTable::findvalues(uint32_t * keys, unsigned no_of_keys, void (*callback)(uint32_t key, uint32_t value)) {
	unsigned no_of_threads = no_of_keys * 32;
	uint32_t * d_keys;
	cudaMalloc(&d_keys, no_of_keys*sizeof(uint32_t));
	cudaMemcpy(d_keys, keys, no_of_keys*sizeof(uint32_t), cudaMemcpyDefault);
	int threads_per_block = 64, blocks = CEILDIV(no_of_threads, threads_per_block);
	utilitykernel::findvalueskernel<<<blocks, threads_per_block>>>(d_keys, no_of_keys, base_slabs, slab_alloc, no_of_buckets, callback);
}

// If some warp divergence bullshit crops up, rewrite this function to have 1 lane
// do all the collation of the found values into an array
__global__ void utilitykernel::findvalueskernel(uint32_t* d_keys, unsigned no_of_keys, Address* base_slabs,
												SlabAlloc* slab_alloc, unsigned no_of_buckets,
												void (*callback)(uint32_t key, uint32_t value)) {
	const int global_warp_id = CEILDIV(blockDim.x, warpSize) * blockIdx.x + (threadIdx.x / warpSize);
	if(global_warp_id < no_of_keys) {
		const uint32_t src_key = d_keys[global_warp_id];
		const unsigned src_bucket = HashFunction::hash(src_key, no_of_buckets);
		Address next = base_slabs[src_bucket];
		while(next != EMPTY_ADDRESS) {
			uint32_t read_data = slab_alloc->ReadSlab(next, __laneID);
			uint32_t next_lane_data = __shfl_down_sync(WARP_MASK, read_data, 1);
			uint32_t found_key_lanes = __ballot_sync(VALID_KEY_MASK, read_data == src_key);
			if(found_key_lanes & 1 << __laneID ) {
				callback(read_data, next_lane_data);
			}
			__syncwarp();
			next = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
		}
	}
}