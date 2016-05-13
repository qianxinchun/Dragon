#include "utils/blocking_queue.hpp"
#include "data_reader.hpp"
#include "blob.hpp"

template<typename T>
BlockingQueue<T>::BlockingQueue() :sync(new Sync()) {}

template<typename T>
void BlockingQueue<T>::push(const T& t){

	//	function_local mutex and unlock automaticly
	//	cause another thread could call pop externally
	//	when this thread is calling push pop&peer at the same time

	boost::mutex::scoped_lock lock(sync->mutex);
	Q.push(t);

	//	must wake one opposite operation avoid deadlock
	//  formula: wait_kind_num = notify_kind_num
	//  referring Producter-Consumer Model and it's semaphore setup method
	sync->condition.notify_one();
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_waiting_msg){
	boost::mutex::scoped_lock lock(sync->mutex);
	while (Q.empty()){
		if (!log_waiting_msg.empty()){ LOG_EVERY_N(INFO, 1000) << log_waiting_msg; }
		sync->condition.wait(lock); //suspend, spare CPU clock
	}
	T t = Q.front();
	Q.pop();
	return t;
}

template<typename T>
T BlockingQueue<T>::peek(){
	boost::mutex::scoped_lock lock(sync->mutex);
	while (Q.empty())
		sync->condition.wait(lock);
	T t = Q.front();
	return t;
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t){
	boost::mutex::scoped_lock lock(sync->mutex);
	if (Q.empty()) return false;
	*t = Q.front();
	Q.pop();
	return true;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t){
	boost::mutex::scoped_lock lock(sync->mutex);
	if (Q.empty()) return false;
	*t = Q.front();
	return true;
}

template<typename T>
size_t BlockingQueue<T>::size(){
	boost::mutex::scoped_lock lock(sync->mutex);
	return Q.size();
}


//	explicit instantiation for Class
//	more info see http://bbs.csdn.net/topics/380250382

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;
template class BlockingQueue < Datum* > ;
template class BlockingQueue < boost::shared_ptr<QueuePair> > ;
