#ifndef HASHFUNCTION_CUH_
#define HASHFUNCTION_CUH_

namespace HashFunction {
	__device__ uint32_t memoryblock_hash(uint32_t global_warp_id, int resident_changes, uint32_t wrap);
	__device__ uint32_t hash(uint32_t value, uint32_t wrap);
}

#endif /* HASHFUNCTION_CUH_ */
