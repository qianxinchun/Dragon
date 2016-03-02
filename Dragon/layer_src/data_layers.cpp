#include "layer_include/data_layers.hpp"
#include "include/math_function.hpp"
#include "data_include/data_transformer.hpp"
template<typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param):
	Layer<Dtype>(param), transform_param(param.transform_param()),has_labels(true){
	//	Layer<Dtype> constructing list:
	//	copy phase/copy blob

}

template<typename Dtype>
void BaseDataLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top){

	//	Non-Labels
	if (top.size() == 1) has_labels = false;
	ptr_transformer.reset(new DataTransformer<Dtype>(transform_param, this->phase));
	ptr_transformer->initRand();
	//	implements in class DataLayer
	dataLayerSetup(bottom, top);
}

template<typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(const LayerParameter& param):
	BaseDataLayer<Dtype>(param){
	//	Blob is not initialized until reshape is called
	//	which can be regarded as a containter in the queue here
	for (int i = 0; i < PREFETCH_COUNT; i++) free.push(&prefetch[i]);
}

template<typename Dtype>
void BasePrefetchingDataLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	BaseDataLayer<Dtype>::layerSetup(bottom, top);
	//	it will apply after calling DataLayer<Dtype>::dataLayerSetup
	//	call mutable_ to malloc SyncedMemory cause reshape just calculate the malloc size
	//	see also void Blob<Dtype>::reshape(vector<int> shape)
	for (int i = 0; i < PREFETCH_COUNT; i++){
		prefetch[i].data.mutable_cpu_data();
		if (has_labels) prefetch[i].label.mutable_cpu_data();
	}
#ifndef CPU_ONLY
	if (Dragon::get_mode() == Dragon::Mode::GPU){
		for (int i = 0; i < PREFETCH_COUNT; i++){
			prefetch[i].data.mutable_gpu_data();
			if (has_labels) prefetch[i].label.mutable_gpu_data();
		}
	}
#endif
	DLOG(INFO) << "Initializing Mutable Prefetch";
	//Transfromer
	startThread();
	DLOG(INFO) << "Prefetch Initialized";
}

template<typename Dtype>
void BasePrefetchingDataLayer<Dtype>::interfaceKernel(){
	//	create GPU async stream
	//	speed up memcpy between CPU and GPU
	//	because cudaMemcpy will be called frequently 
	//	rather than malloc gpu memory firstly(just call cudaMemcpy)
#ifndef CPU_ONLY
	cudaStream_t stream;
	if (Dragon::get_mode() == Dragon::GPU){
		CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	}
#endif
	try{
		while (!must_stop()){
			Batch<Dtype> *batch = free.pop(); //batch has already reshape in dataLayerSetup
			loadBatch(batch); // pure abstract function
#ifndef CPU_ONLY
			if (Dragon::get_mode() == Dragon::GPU){
				batch->data.data()->async_gpu_data(stream);
				// blocking this thread until host->device memcpy finished
				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
#endif
			full.push(batch); //product
		}
	} catch (boost::thread_interrupted&) {}

	//	destroy async stream
#ifndef CPU_ONLY
	if (Dragon::get_mode() == Dragon::GPU) CUDA_CHECK(cudaStreamDestroy(stream));
#endif

}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	// consume
	Batch<Dtype> *batch = full.pop("DataLayer prefectching queue is now empty");
	top[0]->reshapeLike(batch->data);
	dragon_copy<Dtype>(batch->data.count(), top[0]->mutable_cpu_data(), batch->data.cpu_data());
	DLOG(INFO) << "Prefectch copied";
	if (has_labels){
		top[1]->reshapeLike(batch->label);
		dragon_copy(batch->label.count(), top[1]->mutable_cpu_data(), batch->label.cpu_data());
	}
	free.push(batch);
}

//	two parallel threads:	
//	reader thread will start mmediately to buffer Queue<Datum*>
//	prefetching thread will start after all the layerSetup finishing to buffer Queue<Batch*>
template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param) :
	BasePrefetchingDataLayer<Dtype>(param), reader(param) {}

template <typename Dtype>
void DataLayer<Dtype>::dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int batch_size = param.data_param().batch_size();
	//product 
	Datum datum = *(reader.full().peek());
	vector<int> topShape = ptr_transformer->inferBlobShape(datum);
	transformed_data.reshape(topShape);
	topShape[0] = batch_size;
	top[0]->reshape(topShape);
	for (int i = 0; i < PREFETCH_COUNT; i++) prefetch[i].data.reshape(topShape);
	LOG(INFO) << "output data size: (" << top[0]->num() << "," << top[0]->channels() << ","
		<< top[0]->height() << "," << top[0]->width()<<")";
	if (has_labels){
		// 1D size
		topShape = vector<int>(1, batch_size);
		top[1]->reshape(topShape);
		for (int i = 0; i < PREFETCH_COUNT; i++) prefetch[i].label.reshape(topShape);
	}
}

template <typename Dtype>
void DataLayer<Dtype>::loadBatch(Batch<Dtype> *batch){
	// batch has already reshaped in dataLayerSetup
	// check whether it has the blob size
	CHECK(batch->data.count());
	CHECK(transformed_data.count());
	const int batch_size = param.data_param().batch_size();
	// transformed_data keeps 4D size but just regard it as 3D(Image)
	// it will share parts of a batch memory place, and transform directly in a batch
	Dtype *base_data = batch->data.mutable_cpu_data();
	Dtype *base_label = has_labels ? batch->label.mutable_cpu_data() : NULL;
	for (int i = 0; i < batch_size; i++){
		// must refer use '&' to keep data vaild(!!!important)
		Datum &datum = *(reader.full().pop("Waiting for Datum data"));
		int offset = batch->data.offset(i);
		//	share a part of a blob memory 
		transformed_data.set_cpu_data(base_data + offset);
		//	transform datum and copy its value to the part of blob memory
		if (has_labels) base_label[i] = datum.label();
		ptr_transformer->transform(datum, &transformed_data);
		//let the reader to read new datum
		reader.free().push(&datum);
	}
}

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);
INSTANTIATE_CLASS(DataLayer);