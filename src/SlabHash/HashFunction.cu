#include "HashFunction.cuh"
#include <string>
#include <cassert>

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
	ULL data = (((ULL) global_warp_id) << 32) + resident_changes;
	return ::hash(data) % wrap;
}

__device__
uint32_t HashFunction::hash(uint32_t value, uint32_t wrap) {
	return ::hash(value) % wrap;
}

__device__
int HashFunction::unsetbit_index(uint32_t global_warp_id, uint32_t resident_changes, uint32_t value){
	uint32_t flipped_value = ~value;
	int num_unset = __popc(flipped_value);
	if (num_unset == 0) return -1;
	int bit_number = memoryblock_hash(global_warp_id, resident_changes, num_unset);
	int index = 0;
	// TODO: This logic is a bit unclear, can try to make it clearer
	while (true) {
		int numZerosInitially = __clz(flipped_value);
		index += numZerosInitially;
		if (bit_number == 0) break;
		flipped_value <<= numZerosInitially + 1;
		index++;
		bit_number--;
	}
	//Test
	assert ((~value >> (32 - index - 1)) & 1);
	return index;
}
