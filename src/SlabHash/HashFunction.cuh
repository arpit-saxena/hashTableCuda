#ifndef HASHFUNCTION_CUH_
#define HASHFUNCTION_CUH_

#include <cstdint>

namespace HashFunction {
	__device__ uint32_t memoryblock_hash(uint32_t global_warp_id, int resident_changes, uint32_t wrap);
	__device__ uint32_t hash(uint32_t value, uint32_t wrap);
	__device__ int unsetbit_index(uint32_t global_warp_id, uint32_t resident_changes, uint32_t value);
	// ^ Returns the 0-based index of a unset bit hashed by the preceding variables. If all bits are set
	// then returns -1
}

#endif /* HASHFUNCTION_CUH_ */
