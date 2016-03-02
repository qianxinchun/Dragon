#include "layer_include/data_layers.hpp"
#include "data_include/db.hpp"

//	read data serially
template <typename Dtype>
void AppDataLayer<Dtype>::loadData(){
	boost::shared_ptr<DB> db(GetDB(param.data_param().backend()));
	db->Open(param.data_param().source(), DB::READ);
	boost::shared_ptr<Cursor> cursor(db->NewCursor());
	//	read all data 
	while (cursor->valid()){
		Datum* datum = new Datum();
		datum->ParseFromString(cursor->value());
		Q.push(datum);
	}
}

// transform data independently
template <typename Dtype>
void AppDataLayer<Dtype>::transformData(){
	CHECK(batch->data.count());
	CHECK(transformed_data.count());
	Dtype *base_data = batch->data.mutable_cpu_data();
	Dtype *base_label = has_labels ? batch->label.mutable_cpu_data() : NULL;
	const int batch_size = Q.size();
	for (int i = 0; i < batch_size; i++){
		// must refer use '&' to keep data vaild(!!!important)
		Datum &datum = *(Q.front());
		Q.pop();
		int offset = batch->data.offset(i);
		//	share a part of a blob memory 
		transformed_data.set_cpu_data(base_data + offset);
		//	transform datum and copy its value to the part of blob memory
		if (has_labels) base_label[i] = datum.label();
		ptr_transformer->transform(datum, &transformed_data);
	}
}

template <typename Dtype>
void AppDataLayer<Dtype>::dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top){

	// load test data firstly
	loadData();

	const int batch_size = Q.size();
	Datum datum = (*Q.front());
	vector<int> topShape = ptr_transformer->inferBlobShape(datum);
	transformed_data.reshape(topShape);
	//	set the num of data as the batch_size
	topShape[0] = batch_size;
	top[0]->reshape(topShape);
	batch = new Batch<Dtype>();
	batch->data.reshape(topShape);
	if (has_labels){
		// 1D size
		topShape = vector<int>(1, batch_size);
		top[1]->reshape(topShape);
		batch->label.reshape(topShape);
	}
}

template <typename Dtype>
void AppDataLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//	transform data firstly
	transformData();
	top[0]->reshapeLike(batch->data);
	dragon_copy<Dtype>(batch->data.count(), top[0]->mutable_cpu_data(), batch->data.cpu_data());
	if (has_labels){
		top[1]->reshapeLike(batch->label);
		dragon_copy(batch->label.count(), top[1]->mutable_cpu_data(), batch->label.cpu_data());
	}
}

template <typename Dtype>
void AppDataLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//	transform data firstly
	transformData();
	top[0]->reshapeLike(batch->data);
	dragon_gpu_copy(batch->data.count(), top[0]->mutable_gpu_data(), batch->data.gpu_data());
	if (has_labels){
		top[1]->reshapeLike(batch->label);
		dragon_gpu_copy(batch->label.count(), top[1]->mutable_gpu_data(), batch->label.gpu_data());
	}
	// Ensure the copy is synchronous wrt the host, so that the next batch isn't
	// copied in meanwhile.
	CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
}

INSTANTIATE_CLASS(AppDataLayer);