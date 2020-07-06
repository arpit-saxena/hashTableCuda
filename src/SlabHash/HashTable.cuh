#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <cstdint>
#include "SlabAlloc.cuh"

#define ADDRESS_LANE 31
#define EMPTY_KEY (uint32_t)(0xFFFFFFFF)
#define EMPTY_VALUE EMPTY_KEY
#define SEARCH_NOT_FOUND EMPTY_KEY
#define VALID_KEY_MASK (uint32_t)(0x15555555)

typedef unsigned long long ULL;

namespace utilitykernel {
	__global__ void init_table(int, SlabAlloc *, Address *);
	__global__ void findvalueskernel(uint32_t* d_keys, unsigned no_of_keys, 
		Address* base_slabs, SlabAlloc* slab_alloc, unsigned no_of_buckets,
		void (*callback)(uint32_t key, uint32_t value));

	//Sample callback as default
	__device__ void default_callback(uint32_t key, uint32_t value);
}

class HashTable {		// a single object of this will be made on host, and copied to global device memory
		Address * base_slabs;
		SlabAlloc * slab_alloc;
		unsigned no_of_buckets;
	public:
		// Needs to be called with the SlabAlloc pointer pointing to an object placed in device memory
		__host__ HashTable(int size, SlabAlloc * s);
		__host__ ~HashTable();
		__host__ void findvalues(uint32_t * keys, unsigned no_of_keys, void (*callback)(uint32_t key, uint32_t value));

		friend class HashTableOperation;
};

struct Instruction {
	enum Type {
		Insert,
		Delete,
		Search
	};
	
	Type type;
	uint32_t key, value;
};

class HashTableOperation {		// a single object of this will reside on thread-local memory for all threads
	HashTable* hashtable;
	ResidentBlock* resident_block;
	Instruction * instr;
	int laneID;

	bool is_active;
	uint32_t src_key, src_value, read_data;
	Address next;
	int src_lane;


	__device__ static ULL makepair(uint32_t key, uint32_t value);
	__device__ uint32_t ReadSlab(Address slab_addr, int laneID);
	__device__ uint32_t * SlabAddress(Address slab_addr, int laneID);

	__device__ void inserter();
	__device__ void searcher();
	__device__ void deleter();
public:
	__device__ HashTableOperation(Instruction * ins, HashTable * h, ResidentBlock * rb, bool is_active = true);
	__device__ void run();
};

#endif /* HASHTABLE_H */
