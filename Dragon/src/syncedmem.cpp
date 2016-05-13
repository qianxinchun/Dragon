#include "syncedmem.hpp"
void SyncedMemory::to_cpu()
{
	switch (head_){
	case UNINITIALIZED:
		dragonMalloc(&cpu_ptr, size_);
		dragonMemset(cpu_ptr, size_);
		head_ = HEAD_AT_CPU;
		own_cpu_data = true;
		break;
	case HEAD_AT_GPU:
#ifndef CPU_ONLY
		if (cpu_ptr == NULL){
			dragonMalloc(&cpu_ptr, size_);
			own_cpu_data = true;
		}
		dragonGpuMemcpy(cpu_ptr,gpu_ptr,size_);
		head_ = SYNCED;
#endif
		break;
	case HEAD_AT_CPU:
	case SYNCED:
		break;
	}
}

void SyncedMemory::to_gpu()
{
#ifndef CPU_ONLY
	switch (head_){
	case UNINITIALIZED:
		dragonGpuMalloc(&gpu_ptr,size_);
		dragonGpuMemset(gpu_ptr, size_);
		head_ = HEAD_AT_GPU;
		own_gpu_data = true;
		break;
	case HEAD_AT_CPU:
		if (gpu_ptr == NULL){
			dragonGpuMalloc(&gpu_ptr,size_);
			own_gpu_data = true;
		}
		dragonGpuMemcpy(gpu_ptr, cpu_ptr, size_);
		head_ = SYNCED;
		break;
	case HEAD_AT_GPU:
	case SYNCED:
		break;
	}
#endif
}

const void* SyncedMemory::cpu_data(){
	to_cpu();
	return (const void*)cpu_ptr;
}

const void* SyncedMemory::gpu_data(){
	to_gpu();
	return (const void*)gpu_ptr;
}

void SyncedMemory::set_cpu_data(void *data){
	if (own_cpu_data) dragonFree(cpu_ptr);
	cpu_ptr = data;
	head_ = HEAD_AT_CPU;
	own_cpu_data = false;
}

void SyncedMemory::set_gpu_data(void *data){
#ifndef CPU_ONLY
	if (own_gpu_data) dragonGpuFree(gpu_ptr);
	gpu_ptr = data;
	head_ = HEAD_AT_GPU;
	own_gpu_data = false;
#endif
}

void* SyncedMemory::mutable_cpu_data(){
	to_cpu();
	head_ = HEAD_AT_CPU;
	return cpu_ptr;
}

void* SyncedMemory::mutable_gpu_data(){
#ifndef CPU_ONLY
	to_gpu();
	head_ = HEAD_AT_GPU;
	return gpu_ptr;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_data(const cudaStream_t& stream){
	CHECK(head_ == HEAD_AT_CPU);
	//	first allocating memory
	if (gpu_ptr == NULL){
		dragonGpuMalloc(&gpu_ptr, size_);
		own_gpu_data = true;
	}
	const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
	CUDA_CHECK(cudaMemcpyAsync(gpu_ptr, cpu_ptr, size_, kind, stream));
	head_ = SYNCED;
}
#endif

SyncedMemory::~SyncedMemory(){
	if (cpu_ptr && own_cpu_data) dragonFree(cpu_ptr);
#ifndef CPU_ONLY
	if (gpu_ptr && own_gpu_data) dragonGpuFree(gpu_ptr);
#endif
}
