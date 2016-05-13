# ifndef SYNCEDMEM_HPP
# define SYNCEDMEM_HPP

#define NULL 0
#include <cstdlib>
#include <cstring>
#include "common.hpp"

inline void dragonMalloc(void **ptr, size_t size){
	*ptr = malloc(size);
	CHECK(*ptr) << "host allocation of size " << size << " failed";
}
inline void dragonFree(void *ptr){
	free(ptr);
}
inline void dragonMemset(void *ptr,size_t size){
	memset(ptr, 0, size);
}
inline void dragonMemcpy(void* dest, void* src,size_t size){
	memcpy(dest, src, size);
}
#ifndef CPU_ONLY
#include "cuda.h"
inline void dragonGpuMalloc(void **ptr, size_t size){
	CUDA_CHECK(cudaMalloc(ptr, size));
}
inline void dragonGpuFree(void *ptr){
	CUDA_CHECK(cudaFree(ptr));
}
inline void dragonGpuMemset(void *ptr, size_t size){
	CUDA_CHECK(cudaMemset(ptr, 0, size));
}
inline void dragonGpuMemcpy(void *dest, void* src, size_t size){
	CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyDefault));
}
#endif

class SyncedMemory
{
public:
	SyncedMemory():cpu_ptr(NULL), gpu_ptr(NULL), size_(0), head_(UNINITIALIZED) {}
	SyncedMemory(size_t size) :cpu_ptr(NULL), gpu_ptr(NULL), size_(size), head_(UNINITIALIZED) {}
	void to_gpu();
	void to_cpu();
	const void* cpu_data();
	const void* gpu_data();
	void set_cpu_data(void *data);
	void set_gpu_data(void *data);
	void* mutable_cpu_data();
	void* mutable_gpu_data();
#ifndef CPU_ONLY
	void async_gpu_data(const cudaStream_t& stream);
#endif
	enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
	void *cpu_ptr, *gpu_ptr;
	size_t size_;
	bool own_cpu_data, own_gpu_data;
	SyncedHead head_;
	SyncedHead head() { return head_; }
	size_t size() { return size_; }
	~SyncedMemory();
};

# endif