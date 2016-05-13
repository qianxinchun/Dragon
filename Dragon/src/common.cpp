#include "common.hpp"

//	using _getpid() with MSVC,and #include <process.h>
#include <process.h>
#include <ctime>
static boost::thread_specific_ptr<Dragon> thread_instance;

//	for each working thread, allocating independent Dragon Manager
//	Through manager_ptr is static, but each thread shared different static variable
//	this is the specific of boost::thread_specific_ptr 

Dragon& Dragon::Get(){
	if (!thread_instance.get()) thread_instance.reset(new Dragon());
	return *(thread_instance.get());
}

//	generate a random seed according to the current process's pid
int64_t Dragon::cluster_seedgen(){
	int64_t seed, pid, t;
	pid = _getpid();
	t = time(0);
	seed = abs(((t * 181) *((pid - 83) * 359)) % 104729); //set it as you want casually
	return seed;
}

#ifdef CPU_ONLY
//	implements for CPU Manager
Dragon::Dragon():
	mode(Dragon::CPU), solver_count(1), root_solver(true) {}
Dragon::~Dragon() { }
void Dragon::set_device(const int device_id) {}
void Dragon::set_random_seed(const unsigned int seed) {Get().random_generator.reset(new RNG(seed));}
#else
//	implements for CPU/GPU Manager
void Dragon::set_random_seed(const unsigned int seed) {
	// Curand seed
	static bool g_curand_availability_logged = false;
	if (get_curand_generator()) {
		CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(get_curand_generator(), seed));
		CURAND_CHECK(curandSetGeneratorOffset(get_curand_generator(), 0));
	}
	else {
		if (!g_curand_availability_logged) {
			LOG(ERROR) <<"Curand not available. Skipping setting the curand seed.";
			g_curand_availability_logged = true;
		}
	}
	// RNG seed
	Get().random_generator.reset(new RNG(seed));
}
Dragon::Dragon() :
	mode(Dragon::CPU), solver_count(1), root_solver(true),
	cublas_handle(NULL), curand_generator(NULL){
	if (cublasCreate_v2(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
		LOG(ERROR) << "Couldn't create cublas handle.";
	if (curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS
		|| curandSetPseudoRandomGeneratorSeed(curand_generator, cluster_seedgen()) != CURAND_STATUS_SUCCESS)
		LOG(ERROR) << "Couldn't create curand generator.";
}

Dragon::~Dragon(){
	if (cublas_handle) cublasDestroy_v2(cublas_handle);
	if (curand_generator) curandDestroyGenerator(curand_generator);
}

void Dragon::set_device(const int device_id) {
	int current_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	if (current_device == device_id) return;
	// The call to cudaSetDevice must come before any calls to Get, which
	// may perform initialization using the GPU.

	//	reset Device must reset handle and generator???
	CUDA_CHECK(cudaSetDevice(device_id));
	if (Get().cublas_handle) cublasDestroy_v2(Get().cublas_handle);
	if (Get().curand_generator) curandDestroyGenerator(Get().curand_generator);
	cublasCreate_v2(&Get().cublas_handle);
	curandCreateGenerator(&Get().curand_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(Get().curand_generator, cluster_seedgen());
}
#endif
