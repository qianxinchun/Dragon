#include "layer_include/data_layers.hpp"
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	Batch<Dtype> *batch = full.pop("DataLayer prefectching queue is now empty");
	top[0]->reshapeLike(batch->data);
	dragon_gpu_copy(batch->data.count(),top[0]->mutable_gpu_data(), batch->data.gpu_data());

	if (has_labels){
		top[1]->reshapeLike(batch->label);
		dragon_gpu_copy(batch->label.count(), top[1]->mutable_gpu_data(), batch->label.gpu_data());
	}
	// Ensure the copy is synchronous wrt the host, so that the next batch isn't
	// copied in meanwhile.
	CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
	free.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

