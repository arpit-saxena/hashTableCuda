#include <cassert>
#include <string>

#include "HashFunction.cuh"

typedef unsigned long long ULL;

namespace {
// From https://stackoverflow.com/a/12996028/5585431
__device__ ULL hash(ULL x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}
}  // namespace

__device__ uint32_t HashFunction::memoryblock_hash(uint32_t global_warp_id,
                                                   int resident_changes,
                                                   uint32_t wrap) {
  ULL data = (((ULL)global_warp_id) << 32) + resident_changes;
  return ::hash(data) % wrap;
}

__device__ uint32_t HashFunction::hash(uint32_t value, uint32_t wrap) {
  return ::hash(value) % wrap;
}

__device__
    /*
     * Here, the argument bit_number is actually global_warp_id, and index is
     * actually resident_changes These names were chosen to improve readability
     * of this function's implementation as this function is optimized to reuse
     * registers occupied by the function arguments
     */
    int
    HashFunction::unsetbit_index(uint32_t bit_number, uint32_t index,
                                 uint32_t value) {
  value = __brev(~value);
  if (__popc(value) == 0) return -1;
  bit_number = memoryblock_hash(bit_number, index, __popc(value));
  index = 0;
  // TODO: This logic is a bit unclear, can try to make it clearer
  while (true) {
    int numZerosInitially = __clz(value);
    index += numZerosInitially;
    if (bit_number == 0) break;
    value <<= numZerosInitially + 1;
    index++;
    bit_number--;
  }
  // Test
  // assert ((~value >> index) & 1);
  return index;
}
