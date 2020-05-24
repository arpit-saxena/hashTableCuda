#include "HashFunction.cuh"
#include <string>

namespace {
	uint32_t hash(std::string data, uint32_t wrap) {
		// djb2; Taken from http://www.cse.yorku.ca/~oz/hash.html
		unsigned long ret = 5381;
		for (auto c : data) {
			ret = ((ret << 5) + ret) + c; /* hash * 33 + c */
		}

		return ret % wrap;
	}
}

__device__
uint32_t HashFunction::memoryblock_hash(uint32_t global_warp_id, int resident_changes, uint32_t wrap){
	std::string data = global_warp_id;
	data += "/" + resident_changes;
	return hash(data, wrap);
}

__device__
uint32_t HashFunction::hash(uint32_t value, uint32_t wrap) {
	return hash(std::to_string(value), wrap);
}
