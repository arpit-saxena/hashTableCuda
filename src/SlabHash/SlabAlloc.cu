#include <assert.h>
#include <stdio.h>

#include <new>

#include "HashFunction.cuh"
#include "SlabAlloc.cuh"
#include "errorcheck.h"

__device__ unsigned SlabAlloc::numSuperBlocks = 0;
__constant__ SuperBlock *SlabAlloc::initsuperBlocks[initNumSuperBlocks];
__device__ SuperBlock
    *SlabAlloc::dyn_allocated_superBlocks[maxSuperBlocks - initNumSuperBlocks];

__device__ BlockBitMap
    SlabAlloc::bitmaps[maxSuperBlocks * SuperBlock::numMemoryBlocks];

__host__ void SlabAlloc::init() {
  if (initNumSuperBlocks > maxSuperBlocks) {
    // TODO: Better way to handle this?
    printf("Can't allocate %d super blocks. Max is %d", initNumSuperBlocks,
           maxSuperBlocks);
    return;
  }
  int x = initNumSuperBlocks;
  gpuErrchk(cudaMemcpyToSymbol(numSuperBlocks, &x, sizeof(unsigned int)));

  for (int i = 0; i < maxSuperBlocks; i++) {
    SuperBlock *temp = nullptr;
    if (i < initNumSuperBlocks) {
      gpuErrchk(cudaMalloc(&temp, sizeof(SuperBlock)));
      gpuErrchk(cudaMemcpyToSymbol(initsuperBlocks, &temp, sizeof(SuperBlock *),
                                   i * sizeof(SuperBlock *)));
    } else {
      gpuErrchk(cudaMemcpyToSymbol(
          dyn_allocated_superBlocks, &temp, sizeof(SuperBlock *),
          (i - initNumSuperBlocks) * sizeof(SuperBlock *)));
    }
  }

  BlockBitMap *b = nullptr;
  size_t size = 0;
  gpuErrchk(cudaGetSymbolAddress((void **)&b, bitmaps));
  gpuErrchk(cudaGetSymbolSize(&size, bitmaps));
  gpuErrchk(cudaMemset(b, 0, size));
}

__host__ void SlabAlloc::destroy() {
  int size = maxSuperBlocks - initNumSuperBlocks;
  if (size != 0) {
    int threadsPerBlock = 64, numBlocks = CEILDIV(size, threadsPerBlock);
    SlabAlloc::clean_superblocks<<<numBlocks, threadsPerBlock>>>(size);
  }

  for (int i = 0; i < initNumSuperBlocks; i++) {
    SuperBlock *ptr = nullptr;
    gpuErrchk(cudaMemcpyFromSymbol(&ptr, initsuperBlocks, sizeof(SuperBlock *),
                                   i * sizeof(SuperBlock *)));
    if (ptr) gpuErrchk(cudaFree(ptr));
  }
}

__global__ void SlabAlloc::clean_superblocks(const ULL size) {
  int threadID = blockDim.x * blockIdx.x + threadIdx.x;
  while (threadID < size) {
    if (dyn_allocated_superBlocks[threadID])
      free(dyn_allocated_superBlocks[threadID]);
    dyn_allocated_superBlocks[threadID] = nullptr;
    threadID += gridDim.x * blockDim.x;
  }
}

__device__ Address SlabAlloc::makeAddress(uint32_t superBlock_idx,
                                          uint32_t memoryBlock_idx,
                                          uint32_t slab_idx) {
  return (superBlock_idx << (SLAB_BITS + MEMORYBLOCK_BITS)) +
         (memoryBlock_idx << SLAB_BITS) + slab_idx;
}

__device__ SuperBlock *SlabAlloc::getSuperBlockAddr(int index) {
  if (index >= initNumSuperBlocks) {
    index -= initNumSuperBlocks;
    return dyn_allocated_superBlocks[index];
  } else {
    return initsuperBlocks[index];
  }
}

// Currently called with full warp only, so it also assumes full warp
__device__ void SlabAlloc::allocateSuperBlock() {
  assert(__activemask() == WARP_MASK);
  if (__laneID == 0) {
    if (numSuperBlocks < maxSuperBlocks) {
      SuperBlock *newSuperBlock = (SuperBlock *)malloc(sizeof(SuperBlock));
      if (newSuperBlock == nullptr) {
        int OutOfMemory = 0;
        printf("Finally, %d superblocks\n", numSuperBlocks);
        assert(OutOfMemory);
        return;
      }
      SuperBlock *oldSuperBlock =
          (SuperBlock *)atomicCAS((ULL *)(dyn_allocated_superBlocks +
                                          numSuperBlocks - initNumSuperBlocks),
                                  (ULL) nullptr, (ULL)newSuperBlock);
      if (oldSuperBlock != nullptr) {
        free(newSuperBlock);
      } else {
        atomicAdd(&numSuperBlocks, 1);
      }
    }
  }
}

__device__ uint32_t *SlabAlloc::SlabAddress(Address addr, uint32_t laneID) {
  uint32_t slab_idx = addr & ((1 << SLAB_BITS) - 1);
  uint32_t block_idx = (addr >> SLAB_BITS) & ((1 << MEMORYBLOCK_BITS) - 1);
  uint32_t superBlock_idx = (addr >> (SLAB_BITS + MEMORYBLOCK_BITS));
  return (getSuperBlockAddr(superBlock_idx)
              ->memoryBlocks[block_idx]
              .slabs[slab_idx]
              .arr) +
         laneID;
}

__device__ uint32_t SlabAlloc::ReadSlab(Address slab_addr, int laneID) {
  return *(SlabAddress(slab_addr, laneID));
}

__device__ void SlabAlloc::deallocate(
    Address addr) {  // Doesn't need a full warp
  if (__laneID == __ffs(__activemask()) - 1) {
    unsigned global_memory_block_no = addr >> SLAB_BITS;
    unsigned memory_unit_no =
        addr & ((1 << SLAB_BITS) - 1);  // addr%1024, basically
    unsigned lane_no = memory_unit_no / 32, slab_no = memory_unit_no % 32;
    BlockBitMap *resident_bitmap = bitmaps + global_memory_block_no;
    uint32_t *global_bitmap_line = resident_bitmap->bitmap + lane_no;
    atomicAnd(global_bitmap_line, ~(1u << slab_no));
  }
  // TODO Check for divergence here
}

__device__ ResidentBlock::ResidentBlock() {
  resident_changes = -1;
  set();
}

// Needs full warp
__device__ void ResidentBlock::set() {
  if (resident_changes % max_resident_changes == 0 && resident_changes != 0) {
    SlabAlloc::allocateSuperBlock();
#ifndef NDEBUG
    if (__laneID == 0)  // DEBUG
      printf(
          "\tset()->allocateSuperBlock() called by set(), "
          "resident_changes=%d\n",
          resident_changes);
#endif  // !NDEBUG
    // resident_changes = -1;	// So it becomes 0 after a memory block is found
  }
  // unsigned memory_block_no = HashFunction::memoryblock_hash(__global_warp_id,
  // resident_changes, SuperBlock::numMemoryBlocks);
  uint32_t super_memory_block_no = HashFunction::memoryblock_hash(
      __global_warp_id, resident_changes,
      SlabAlloc::numSuperBlocks *
          SuperBlock::numMemoryBlocks /*total_memory_blocks*/);
#ifndef NDEBUG
  // if (__laneID == 0 && resident_changes != -1)		//DEBUG
//		printf("\tset()->super_memory_block_no=hash(__global_warp_id=%d,
//resident_changes=%d, total_memory_blocks=%d)=%d\n", __global_warp_id,
//resident_changes, slab_alloc->getNumSuperBlocks() *
//SuperBlock::numMemoryBlocks, super_memory_block_no);
#endif  // !NDEBUG

  starting_addr = super_memory_block_no << SLAB_BITS;
  ++resident_changes;
  BlockBitMap *resident_bitmap = SlabAlloc::bitmaps + super_memory_block_no;
  resident_bitmap_line = resident_bitmap->bitmap[__laneID];
}

__device__ Address ResidentBlock::warp_allocate() {
  // TODO remove this loop maybe
  const int max_local_rbl_changes = max_resident_changes;
  int memoryblock_changes = 0;
  for (int local_rbl_changes = 0; local_rbl_changes <= max_local_rbl_changes;
       ++local_rbl_changes) {  // review the loop termination condition
    int slab_no, allocator_thread_no;
    while (true) {  // Review this loop
      slab_no = HashFunction::unsetbit_index(
          __global_warp_id, local_rbl_changes, resident_bitmap_line);
      allocator_thread_no =
          HashFunction::unsetbit_index(__global_warp_id, local_rbl_changes,
                                       ~__ballot_sync(WARP_MASK, slab_no + 1));
      if (allocator_thread_no ==
          -1) {  // All memory units are full in the memory block
        const int max_allowed_memoryblock_changes =
            2 /*max_allowed_superblock_changes*/ * max_resident_changes;
        if (memoryblock_changes > max_allowed_memoryblock_changes) {
          int khela = 0;
          assert(khela);
        }
        set();
        ++memoryblock_changes;
      } else {
        break;
      }
    }

    Address allocated_address = EMPTY_ADDRESS;
    if (__laneID == allocator_thread_no) {
      uint32_t i = 1 << slab_no;
      auto global_memory_block_no = starting_addr >> SLAB_BITS;
      BlockBitMap *resident_bitmap =
          SlabAlloc::bitmaps + global_memory_block_no;
      uint32_t *global_bitmap_line = resident_bitmap->bitmap + __laneID;
      auto oldval = atomicOr(global_bitmap_line, i);
      resident_bitmap_line = oldval | i;
      if ((oldval & i) == 0) {
        allocated_address = starting_addr + (__laneID << 5) + slab_no;
      }
    }

    __syncwarp();
    Address toreturn =
        __shfl_sync(WARP_MASK, allocated_address, allocator_thread_no);
    if (toreturn != EMPTY_ADDRESS) {
      *(SlabAlloc::SlabAddress(toreturn, __laneID)) = EMPTY_ADDRESS;
      return toreturn;
    }
    // TODO check for divergence on this functions return
  }
  // This means all max_local_rbl_changes attempts to allocate memory failed as
  // the atomicCAS call kept failing Terminate
  int mahakhela = 0;
  assert(mahakhela);

  return EMPTY_ADDRESS;  // Will never execute
}

#ifndef NDEBUG
__device__ Address ResidentBlock::warp_allocate(int *x) {  // DEBUG
  __shared__ int lrc[32][8];
  __shared__ int sn[32][8];
  __shared__ int atn[32][8];
  __shared__ uint32_t ov[32][8];
  __syncwarp();
  int warp_id_in_block = threadIdx.x / warpSize;
  for (int i = 0; i < 8; ++i) lrc[warp_id_in_block][i] = -1;
  // TODO remove this loop maybe
  const int max_local_rbl_changes = max_resident_changes;
  int memoryblock_changes = 0;
  for (/*int local_rbl_changes = 0*/ *x = 0;
       /*local_rbl_changes*/ *x <= max_local_rbl_changes;
       ++(*x) /*++local_rbl_changes*/) {  // review the loop termination
                                          // condition
    int slab_no, allocator_thread_no;
    auto local_rbl_changes = *x;
    while (true) {  // Review this loop
      slab_no = HashFunction::unsetbit_index(
          __global_warp_id, local_rbl_changes, resident_bitmap_line);
      allocator_thread_no =
          HashFunction::unsetbit_index(__global_warp_id, local_rbl_changes,
                                       ~__ballot_sync(WARP_MASK, slab_no + 1));
      if (allocator_thread_no ==
          -1) {  // All memory units are full in the memory block
        const int max_allowed_memoryblock_changes =
            2 /*max_allowed_superblock_changes*/ * max_resident_changes;
        if (memoryblock_changes > max_allowed_memoryblock_changes) {
          int khela = 0;
          assert(khela);
        }
        __syncwarp();
        // if (__laneID == 0)
        // printf("Warp ID=%d, local_rbl_changes=%d, memoryblock_changes=%d,
        // called set()\n", __global_warp_id, *x, memoryblock_changes);
        set();
        ++memoryblock_changes;
      } else {
        break;
      }
    }

    Address allocated_address = EMPTY_ADDRESS;
    if (__laneID == allocator_thread_no) {
      uint32_t i = 1 << slab_no;
      auto global_memory_block_no = starting_addr >> SLAB_BITS;
      BlockBitMap *resident_bitmap =
          SlabAlloc::bitmaps + global_memory_block_no;
      uint32_t *global_bitmap_line = resident_bitmap->bitmap + __laneID;
      auto oldval = atomicOr(global_bitmap_line, i);
      resident_bitmap_line = oldval | i;
      if ((oldval & i) == 0) {
        allocated_address = starting_addr + (__laneID << 5) + slab_no;
      } else {
        lrc[warp_id_in_block][*x] = *x;
        sn[warp_id_in_block][*x] = slab_no;
        atn[warp_id_in_block][*x] = allocator_thread_no;
        ov[warp_id_in_block][*x] = oldval;
      }
    }

    __syncwarp();
    Address toreturn =
        __shfl_sync(WARP_MASK, allocated_address, allocator_thread_no);
    if (toreturn != EMPTY_ADDRESS) {
      // uint32_t* ptr = SlabAlloc::SlabAddress(toreturn, __laneID);
      *(SlabAlloc::SlabAddress(toreturn, __laneID)) = EMPTY_ADDRESS;
      return toreturn;
    }
    // TODO check for divergence on this functions return
  }
  // This means all max_local_rbl_changes attempts to allocate memory failed as
  // the atomicCAS call kept failing Terminate
  /*SlabAlloc::status = 2;
  __threadfence();
  int mahakhela = 0;
  assert(mahakhela);
  asm("trap;");*/
  __syncwarp();
  if (__laneID == 0) {
    printf(
        "warp_allocate() failed for Warp ID=%d. Details of each iteration:\n",
        __global_warp_id);
    for (int i = 0; i < 8; ++i) {
      if (lrc[warp_id_in_block][i] != -1)
        printf(
            "-> Warp ID=%d, local_rbl_changes=%d, oldval=%x, slab_no=%d, "
            "allocator_thread_no=%d\n",
            __global_warp_id, lrc[warp_id_in_block][i], ov[warp_id_in_block][i],
            sn[warp_id_in_block][i], atn[warp_id_in_block][i]);
    }
    printf(
        "----------------------------------------------------------------------"
        "---------------------------------\n");
  }
  return EMPTY_ADDRESS;  // Will never execute
}
#endif  // !NDEBUG
