 #ifndef DATA_READER_HPP
#define DATA_READER_HPP
#include "include/dragon.pb.h"
#include "include/blocking_queue.hpp"
#include "include/dragon_thread.hpp"
#include "data_include/db.hpp"
// QueuePair is the basic applicated DataStructure
// it is equal to a producter/consumer model
// free&full can be regard as a memory queue with Semaphore

class QueuePair{
public:
	QueuePair(const int size);
	~QueuePair();
	BlockingQueue<Datum*> free; // as producter queue
	BlockingQueue<Datum*> full; // as consumer queue
};

// Body is a basic data-reading thread
// which re-write the virtual function --> void interfaceKernel()
// it will read Datum directly and circularly from LMDB
// until C++ Destructor has triggered
// actually thread will be blocked at most time

class Body :public DragonThread{
public:
	Body(const LayerParameter& param);
	virtual ~Body();
	BlockingQueue<boost::shared_ptr<QueuePair> > new_pairs;
protected:
	void interfaceKernel();
	void read_one(Cursor *cursor, QueuePair *pair);
	LayerParameter param;

};
class DataReader
{
public:
	DataReader(const LayerParameter& param);
	BlockingQueue<Datum*>& free() const  { return ptr_pair->free; }
	BlockingQueue<Datum*>& full() const  { return ptr_pair->full; }
	~DataReader();
	static string source_key(const LayerParameter& param){
		return param.name() + ":" + param.data_param().source();
	}
private:
	LayerParameter param;
	boost::shared_ptr<QueuePair> ptr_pair;
	boost::shared_ptr<Body> ptr_body;
	static map<string, boost::weak_ptr<Body> > global_bodies;
};



#endif

