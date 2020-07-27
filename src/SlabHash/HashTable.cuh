#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <cstdint>
#include "SlabAlloc.cuh"

#define ADDRESS_LANE 31
#define EMPTY_KEY (uint32_t)(0xFFFFFFFF)
#define EMPTY_VALUE EMPTY_KEY
#define SEARCH_NOT_FOUND EMPTY_KEY
#define VALID_KEY_MASK (uint32_t)(0x15555555)

#define THREADS_PER_BLOCK 64

typedef unsigned long long ULL;

namespace utilitykernel {
	__global__ void init_table(int, Address *);
	__global__ void findvalueskernel(uint32_t* d_keys, unsigned no_of_keys, 
		Address* base_slabs, unsigned no_of_buckets,
		void (*callback)(uint32_t key, uint32_t value));
}

class HashTable {		// a single object of this will be made on host, and copied to global device memory
		Address * base_slabs;
		const unsigned no_of_buckets;
	public:
		// Needs to be called with the SlabAlloc pointer pointing to an object placed in device memory
		__host__ HashTable(int size);
		__host__ ~HashTable();
		__host__ void findvalues(uint32_t * keys, unsigned no_of_keys, void (*callback)(uint32_t key, uint32_t value));

		friend class HashTableOperation;
};

struct Instruction {
	enum Type {
		Insert,
		Delete
	};
	
	Type type;
	uint32_t key, value;
};

class HashTableOperation {		// a single object of this will reside on thread-local memory for all threads
	const HashTable* __restrict__ const hashtable;
	ResidentBlock* const __restrict__ resident_block;

	__device__ static ULL makepair(uint32_t key, uint32_t value);
	__device__ uint32_t ReadSlab(Address slab_addr, int laneID);
	__device__ uint32_t * SlabAddress(Address slab_addr, int laneID);

	__device__ void inserter(uint32_t s_read_data[], uint32_t src_key, uint32_t src_value, int src_lane, uint32_t &work_queue, Address &next);
	//__device__ void searcher(uint32_t s_read_data[], uint32_t src_key, int src_lane, uint32_t &work_queue, Address &next);
	__device__ void deleter(uint32_t s_read_data[], uint32_t src_key, uint32_t src_value, int src_lane, uint32_t &work_queue, Address &next);
public:
	__device__ HashTableOperation(const HashTable * const __restrict__ h, ResidentBlock * const __restrict__ rb);
	__device__ void run(const Instruction::Type type, const uint32_t key, uint32_t value, bool is_active = true);
};

#endif /* HASHTABLE_H */
