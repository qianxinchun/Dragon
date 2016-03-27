#include "include/dragon_thread.hpp"
#include "direct.h"
#include "iostream"
using namespace std;
//	parameters list tranfers from parent thread(main thread)
//	refer this function when create a boost::thread(child thread)
//	get-->set is not a repeated action, get_func called by parent thread
//	where set_func called by children thread, they sharing different Dragon Manager

void DragonThread::initializeThread(int device, Dragon::Mode mode, int rand_seed, int solver_count, bool root_solver){
#ifndef CPU_ONLY
	CUDA_CHECK(cudaSetDevice(device));
#endif
	Dragon::set_random_seed(rand_seed);
	Dragon::set_mode(mode);
	Dragon::set_solver_count(solver_count);
	Dragon::set_root_solver(root_solver);
	interfaceKernel();  //do nothing
}

//	called by main thread
//	using main thread's configurations
//	after that , following I/O works transfer to child threads
void DragonThread::startThread(){
	CHECK(!is_start());
	int device = 0;
#ifndef CPU_ONLY
	CUDA_CHECK(cudaGetDevice(&device));
#endif
	Dragon::Mode mode = Dragon::get_mode();
	unsigned int seed = Dragon::get_random_value();
	int solver_count = Dragon::get_solver_count();
	bool root_solver = Dragon::get_root_solver();
	try{
		thread.reset(new boost::thread(&DragonThread::initializeThread,
							this, device, mode, seed, solver_count, root_solver));
	}
	catch (std::exception& e){ LOG(FATAL) << "Thread exception: " << e.what(); }

	//	<boost::thread> will start immediately
	//	if the main thread(main function) finished after that when debuging
	//	you will think that thread is not start , that's wrong because main thread is done
	//	and child thread doom to be destroyed
}

void DragonThread::stopThread(){
	if (is_start()){
		thread->interrupt();
	}
	//get all CPU resources to stop immediately ???
	try{thread->join();}
	catch (boost::thread_interrupted&) {}
	catch (std::exception& e){ LOG(FATAL) << "Thread exception: " << e.what(); }
}

bool DragonThread::is_start(){
	return thread&&thread->joinable();
}

bool DragonThread::must_stop(){

	//return true once call thread->interrupt()
	//break Reading-LOOP and complete the thread's working function
	return boost::this_thread::interruption_requested();
}

DragonThread::~DragonThread(){
	stopThread();
}
