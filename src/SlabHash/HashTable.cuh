#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <cstdint>
#include "SlabAlloc.cuh"

typedef unsigned long long ULL;

namespace hashtbl {
	__global__ void init_table(int, SlabAlloc *, Address *);
	void init_consts();
	__constant__ __device__ extern uint32_t EMPTY_KEY, EMPTY_VALUE, SEARCH_NOT_FOUND, VALID_KEY_MASK, WARP_MASK;
	__constant__ __device__ extern Address EMPTY_ADDRESS;
}

class HashTable {		// a single object of this will be made on host, and copied to global device memory
		Address * base_slabs;
		SlabAlloc * slab_alloc;
		unsigned no_of_buckets;
	public:
		HashTable(int size, SlabAlloc * s);
		friend class HashTableOperation;
};

struct Instruction {
	enum Type {
		Insert,
		Delete,
		Search,
		FindAll
	};
	
	Type type;
	uint32_t key, value;
	uint32_t * foundvalues;
};

class HashTableOperation {		// a single object of this will reside on thread-local memory for all threads
	HashTable* hashtable;
	ResidentBlock* resident_block;
	Instruction instr;
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
	__device__ void finder();
public:
	__device__ void init(HashTable * h, ResidentBlock * rb, Instruction ins);
	__device__ void run();
};

#endif /* HASHTABLE_H */
