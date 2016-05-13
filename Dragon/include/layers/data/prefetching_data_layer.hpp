# ifndef PREFETCHING_DATA_LAYERS_HPP
# define PREFETCHING_DATA_LAYERS_HPP

#include "../../dragon_thread.hpp"
#include "base_data_layer.hpp"

template<typename Dtype>
class BasePrefetchingDataLayer :public BaseDataLayer<Dtype>, public DragonThread {
public:
	BasePrefetchingDataLayer(const LayerParameter& param);
	~BasePrefetchingDataLayer(){
		stopThread();
		Batch<Dtype> *batch;
		while (free.try_pop(&batch));
		while (full.try_pop(&batch));
		delete[] prefetch;
	}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	const int PREFETCH_COUNT;
	DataReader reader;
protected:
	virtual void interfaceKernel();
	virtual void loadBatch(Batch<Dtype>* batch);
	Batch<Dtype>* prefetch;
	BlockingQueue<Batch<Dtype>*> free;
	BlockingQueue<Batch<Dtype>*> full;
};

# endif