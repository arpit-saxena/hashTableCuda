#include "HashFunction.cuh"
#include <string>

typedef unsigned long long ULL;

namespace {
    // From https://stackoverflow.com/a/12996028/5585431
	__device__ ULL hash(ULL x) {
	    x = ((x >> 16) ^ x) * 0x45d9f3b;
	    x = ((x >> 16) ^ x) * 0x45d9f3b;
	    x = (x >> 16) ^ x;
	    return x;
	}
}

__device__
uint32_t HashFunction::memoryblock_hash(uint32_t global_warp_id, int resident_changes, uint32_t wrap){
	ULL data = ((ULL) global_warp_id) << 32 + resident_changes;
	return ::hash(data) % wrap;
}

__device__
uint32_t HashFunction::hash(uint32_t value, uint32_t wrap) {
	return ::hash(value) % wrap;
}
