#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <cstdint>
#include "SlabAlloc.cuh"

#define ADDRESS_LANE 31
#define EMPTY_KEY (uint32_t)(0xFFFFFFFF)
#define EMPTY_VALUE EMPTY_KEY
#define SEARCH_NOT_FOUND EMPTY_KEY
#define VALID_KEY_MASK (uint32_t)(0xAAAAAAA8)

typedef unsigned long long ULL;

namespace utilitykernel {
	__global__ void init_table(int, SlabAlloc *, Address *);
}

class HashTable {		// a single object of this will be made on host, and copied to global device memory
		Address * base_slabs;
		SlabAlloc * slab_alloc;
		unsigned no_of_buckets;
	public:
		__host__ HashTable(int size, SlabAlloc * s);
		__host__ ~HashTable();
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
	uint32_t * foundvalues = nullptr;		//Will be set to point to an array in global memory by finder

	__device__ ~Instruction();
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
	__device__ HashTableOperation(HashTable * h, ResidentBlock * rb, Instruction ins);
	__device__ void run();
};

#endif /* HASHTABLE_H */
