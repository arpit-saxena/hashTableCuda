#include <assert.h>
#include <stdio.h>

#include "HashFunction.cuh"
#include "HashTable.cuh"

__host__ HashTable::HashTable(int size) : no_of_buckets(size) {
  gpuErrchk(cudaMalloc(&base_slabs, no_of_buckets * sizeof(Address)));
  int threads_per_block = THREADS_PER_BLOCK /* warp size*2 */,
      blocks = no_of_buckets / (threads_per_block / 32);
  utilitykernel::init_table<<<blocks, threads_per_block>>>(no_of_buckets,
                                                           base_slabs);
}

__global__ void utilitykernel::init_table(int no_of_buckets,
                                          Address *base_slabs) {
  ResidentBlock rb;
  int warp_id = __global_warp_id;
  while (warp_id < no_of_buckets) {
    base_slabs[warp_id] = rb.warp_allocate();
    warp_id += __numwarps_in_block * gridDim.x * gridDim.y * gridDim.z;
  }
}

__host__ HashTable::~HashTable() { gpuErrchk(cudaFree(base_slabs)); }

__device__ ULL HashTableOperation::makepair(uint32_t key, uint32_t value) {
  uint32_t pair[] = {key, value};
  return *reinterpret_cast<ULL *>(pair);
}

__device__ HashTableOperation::HashTableOperation(
    const HashTable *const __restrict__ h, ResidentBlock *const __restrict__ rb)
    : hashtable(h), resident_block(rb) {}

__device__ void HashTableOperation::run(const Instruction::Type type,
                                        const uint32_t key, uint32_t value,
                                        bool is_active) {
  static const int warp_size = 32;
  __shared__ uint32_t s_read_data[THREADS_PER_BLOCK];
  __shared__ uint32_t s_src_key[THREADS_PER_BLOCK / warp_size];
  __shared__ uint32_t s_src_value[THREADS_PER_BLOCK / warp_size];
  __shared__ uint32_t s_work_queue[THREADS_PER_BLOCK / warp_size];
  __shared__ uint32_t s_old_work_queue[THREADS_PER_BLOCK / warp_size];
  __shared__ Address s_next[THREADS_PER_BLOCK / warp_size];
  __shared__ int s_src_lane[THREADS_PER_BLOCK / warp_size];
  __shared__ Instruction::Type s_src_instrtype[THREADS_PER_BLOCK / warp_size];

  uint32_t &src_key = s_src_key[__local_warp_id],
           &src_value = s_src_value[__local_warp_id],
           &work_queue = s_work_queue[__local_warp_id],
           &old_work_queue = s_old_work_queue[__local_warp_id];
  Address &next = s_next[__local_warp_id];
  int &src_lane = s_src_lane[__local_warp_id];
  Instruction::Type &src_instrtype = s_src_instrtype[__local_warp_id];

  work_queue = __ballot_sync(WARP_MASK, is_active), old_work_queue = 0;
  while (work_queue != 0) {
    if (work_queue != old_work_queue) {
      src_lane = __ffs(work_queue);
      assert(src_lane >= 1 && src_lane <= 32);
      --src_lane;
      src_instrtype = static_cast<Instruction::Type>(
          __shfl_sync(WARP_MASK, type, src_lane));
      src_key = __shfl_sync(WARP_MASK, key, src_lane);
      src_value = __shfl_sync(WARP_MASK, value, src_lane);
      unsigned src_bucket =
          HashFunction::hash(src_key, hashtable->no_of_buckets);
      next = hashtable->base_slabs[src_bucket];
      old_work_queue = work_queue;
    }
    s_read_data[__block_threadIdx] = SlabAlloc::ReadSlab(next, __laneID);
    /*if (next == 29299) {
            printf("slab 29299 has %d at %d\n", s_read_data[__block_threadIdx],
    __laneID);
    }*/
    switch (src_instrtype) {
      case Instruction::Insert:
        inserter(s_read_data, src_key, src_value, src_lane, work_queue, next);
        break;
      case Instruction::Delete:
        deleter(s_read_data, src_key, src_value, src_lane, work_queue, next);
        break;
        /*case Instruction::Search:
                searcher(s_read_data, src_key, src_lane, work_queue, next);
                break;*/
    }
  }
}

/*__device__ void HashTableOperation::searcher(uint32_t s_read_data[], uint32_t
src_key, int src_lane, uint32_t &work_queue, Address &next) { auto found_lane =
__ffs(__ballot_sync(VALID_KEY_MASK, s_read_data[__block_threadIdx] == src_key));
        if(__laneID == src_lane) {
                if(found_lane != 0) {
                        --found_lane;
                        uint32_t found_value =
s_read_data[__local_warp_id*warpSize + found_lane + 1]; instr->value =
found_value; work_queue &= ~((uint32_t)(1<<__laneID));
                }
                else{
                        auto next_ptr = s_read_data[__local_warp_id*warpSize +
ADDRESS_LANE]; if(next_ptr == EMPTY_ADDRESS) { instr->value = SEARCH_NOT_FOUND;
                                work_queue &= ~((uint32_t)(1<<__laneID));
                        }
                        else{
                                next = next_ptr;
                        }
                }
        }
}*/

__device__ __forceinline__ void HashTableOperation::inserter(
    uint32_t s_read_data[], uint32_t src_key, uint32_t src_value, int src_lane,
    uint32_t &work_queue, Address &next) {
  auto dest_lane = __ffs(__ballot_sync(
      VALID_KEY_MASK, s_read_data[__block_threadIdx] == EMPTY_KEY));
  if (dest_lane != 0) {
    --dest_lane;
    if (__laneID == src_lane) {
      auto empty_pair = makepair(EMPTY_KEY, EMPTY_VALUE);
      auto old_pair = atomicCAS((ULL *)SlabAlloc::SlabAddress(next, dest_lane),
                                empty_pair, makepair(src_key, src_value));
      if (old_pair == empty_pair) {
        work_queue &= ~((uint32_t)(1 << __laneID));
      }
    }
  } else {
    auto next_ptr = s_read_data[__local_warp_id * warpSize + ADDRESS_LANE];
    if (next_ptr == EMPTY_ADDRESS) {
      auto new_slab_ptr = resident_block->warp_allocate();
      if (__laneID == ADDRESS_LANE) {
        auto oldptr = atomicCAS(SlabAlloc::SlabAddress(next, __laneID),
                                EMPTY_ADDRESS, new_slab_ptr);
        if (oldptr != EMPTY_ADDRESS) {
          SlabAlloc::deallocate(new_slab_ptr);
          new_slab_ptr = oldptr;
        }
        next = new_slab_ptr;
      }
    } else {
      next = next_ptr;
    }
  }
}

__device__ __forceinline__ void HashTableOperation::deleter(
    uint32_t s_read_data[], uint32_t src_key, uint32_t src_value, int src_lane,
    uint32_t &work_queue, Address &next) {
  auto found_lane = __ffs(__ballot_sync(
      VALID_KEY_MASK, s_read_data[__block_threadIdx] == src_key &&
                          s_read_data[__block_threadIdx + 1] == src_value));
  if (__laneID == src_lane) {
    if (found_lane != 0) {
      --found_lane;
      auto existing_pair = makepair(src_key, src_value);
      auto old_pair =
          atomicCAS((ULL *)SlabAlloc::SlabAddress(next, found_lane),
                    existing_pair, makepair(EMPTY_KEY, EMPTY_VALUE));
      if (old_pair == existing_pair) {
        work_queue &= ~((uint32_t)(1 << __laneID));
      }
    } else {
      auto next_ptr = s_read_data[__local_warp_id * warpSize + ADDRESS_LANE];
      if (next_ptr == EMPTY_ADDRESS) {
        work_queue &= ~((uint32_t)(1 << __laneID));
      } else {
        next = next_ptr;
      }
    }
  }
}

__host__ __device__ void HashTable::findvalues(
    uint32_t *d_keys, unsigned no_of_keys,
    void (*callback)(uint32_t key, uint32_t value), cudaStream_t stream) {
  unsigned no_of_threads = no_of_keys * 32;
  int threads_per_block = THREADS_PER_BLOCK,
      blocks = CEILDIV(no_of_threads, threads_per_block);
  utilitykernel::findvalueskernel<<<blocks, threads_per_block, 0, stream>>>(
      d_keys, no_of_keys, base_slabs, no_of_buckets, callback, this);
}

// If some warp divergence bullshit crops up, rewrite this function to have 1
// lane do all the collation of the found values into an array
__global__ void utilitykernel::findvalueskernel(
    uint32_t *d_keys, unsigned no_of_keys, Address *base_slabs,
    unsigned no_of_buckets, void (*callback)(uint32_t key, uint32_t value),
    HashTable *table) {
  for (int i = __global_warp_id; i < no_of_keys; i += __numwarps_in_block) {
    table->findvalue(d_keys[__global_warp_id], callback);
  }
}

__device__ void HashTable::findvalue(uint32_t key,
                                     void (*callback)(uint32_t key,
                                                      uint32_t value)) {
  const unsigned src_bucket = HashFunction::hash(key, no_of_buckets);
  Address next = base_slabs[src_bucket];
  while (next != EMPTY_ADDRESS) {
    uint32_t read_data = SlabAlloc::ReadSlab(next, __laneID);
    // printf("next: %x -> %x, key -> %d, laneid -> %d\n", next, read_data, key,
    // threadIdx.y);
    uint32_t next_lane_data = __shfl_down_sync(WARP_MASK, read_data, 1);
    uint32_t found_key_lanes = __ballot_sync(VALID_KEY_MASK, read_data == key);
    if (found_key_lanes & 1 << __laneID) {
      callback(read_data, next_lane_data);
    }
    __syncwarp();
    next = __shfl_sync(WARP_MASK, read_data, ADDRESS_LANE);
  }
}