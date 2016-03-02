#ifndef BLOCKING_QUEUE
#define BLOCKING_QUEUE
#include "string"
#include "queue"
#include "include/common.hpp"
using namespace std;
template <typename T>
class BlockingQueue
{
public:
	BlockingQueue();
	void push(const T& t); 
	T pop(const string& log_waiting_msg="");
	T peek();
	size_t size();

	// try_func return false when need blocking
	// try_func for destructor
	bool try_pop(T* t);
	bool try_peek(T* t);
	class Sync{
	public:
		mutable boost::mutex mutex;
		boost::condition_variable condition;
	};
private:
	queue<T> Q;
	boost::shared_ptr<Sync> sync;
};


#endif

