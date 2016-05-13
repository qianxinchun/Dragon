#include "data_reader.hpp"

map<string, boost::weak_ptr<Body> > DataReader::global_bodies;

static boost::mutex bodies_mutex;
DataReader::DataReader(const LayerParameter& param){
	ptr_pair.reset(new QueuePair(
		param.data_param().prefech()*param.data_param().batch_size()));
	boost::mutex::scoped_lock lock(bodies_mutex);
	string hash_key = source_key(param);
	boost::weak_ptr<Body> weak = global_bodies[hash_key];
	ptr_body = weak.lock();
	if (!ptr_body){
		ptr_body.reset(new Body(param));
		global_bodies[hash_key] = boost::weak_ptr<Body>(ptr_body);
	}
	ptr_body->new_pairs.push_back(ptr_pair);
}


DataReader::~DataReader(){
	string hash_key = source_key(param);
	//	release internal body thread

	ptr_body.reset();
	boost::mutex::scoped_lock lock(bodies_mutex);
    //  if released successfully, then remove the key from global_bodies
	if (global_bodies[hash_key].expired()) global_bodies.erase(hash_key);
}

QueuePair::QueuePair(const int size){
	// set the upbound for a producter
	for (int i = 0; i < size; i++) free.push(new Datum());
}

QueuePair::~QueuePair(){
	// release and clear
	Datum *datum;
	while (free.try_pop(&datum)) delete datum;
	while (full.try_pop(&datum)) delete datum;
}

Body::Body(const LayerParameter& param) :param(param){
	//	start reading immediately when constructor complete 
	//	it is async comparing with main thread and blob-making thread
	startThread();
}

Body::~Body() {
	// stop reading 
	//force_stop = true;
	stopThread();
}

void Body::read_one(Cursor *cursor, QueuePair *pair){
	//	could block here when pre-buffer enough Datum
	Datum *datum = pair->free.pop();
	//	LMDB<string,string>
	//	Google Buffer Protocol can decode string
	//	it can be done much quicker than SQL method(e.g. SQLite)
	datum->ParseFromString(cursor->value());
	pair->full.push(datum);
	cursor->Next();
	//	until stop training, we need read data circularly
	if (!cursor->valid()){
		DLOG(INFO) << "Restarting data prefeching from start.\n";
		cursor->SeekToFirst();
	}
}

//	re-write for specific task: reading datum from LMDB
void Body::interfaceKernel(){
	boost::shared_ptr<DB> db(GetDB(param.data_param().backend()));
	db->Open(param.data_param().source(), DB::READ);
	boost::shared_ptr<Cursor> cursor(db->NewCursor());
	try{
		//	default solver_count=1
		int solver_count = param.phase() == TRAIN ? Dragon::get_solver_count() : 1;
		//	working period
		while (!must_stop()){
			for (int i = 0; i < solver_count; i++) 
				read_one(cursor.get(), new_pairs[i].get());
		}
		//  complex condition
	} catch (boost::thread_interrupted&) {}
}
