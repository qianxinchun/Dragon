#ifndef DATA_LAYERS_HPP
#define DATA_LAYERS_HPP
#include "layer_include/layer.hpp"
#include "include/dragon_thread.hpp"
#include "include/blocking_queue.hpp"
#include "include/common.hpp"
#include "data_include/data_reader.hpp"
#include "data_include/data_transformer.hpp"
template<typename Dtype>
class BaseDataLayer:public Layer<Dtype>
{
public:
	BaseDataLayer(const LayerParameter& param);
	~BaseDataLayer() {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top)=0;
protected:
	TransformationParameter transform_param;
	boost::shared_ptr< DataTransformer<Dtype> > ptr_transformer;
	Blob<Dtype> transformed_data;
	bool has_labels;
};

template<typename Dtype>
class Batch{
public:
	Blob<Dtype> data, label;
	~Batch() { }
};

template<typename Dtype>
class BasePrefetchingDataLayer :public BaseDataLayer<Dtype>,public DragonThread {
public:
	BasePrefetchingDataLayer(const LayerParameter& param);
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	static const int PREFETCH_COUNT = 4;
protected:
	virtual void interfaceKernel();
	virtual void loadBatch(Batch<Dtype>* batch) = 0;
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	Blob<Dtype> transformed_data;
	Batch<Dtype> prefetch[PREFETCH_COUNT];
	BlockingQueue<Batch<Dtype>*> free;
	BlockingQueue<Batch<Dtype>*> full;
};

template<typename Dtype>
class DataLayer :public BasePrefetchingDataLayer < Dtype > {
public:
	DataLayer(const LayerParameter& param);
	~DataLayer() { this->stopThread(); }
	void dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	void loadBatch(Batch<Dtype>* batch);
	DataReader reader;
};

template<typename Dtype>
class AppDataLayer :public BaseDataLayer < Dtype >{
public:
	AppDataLayer(const LayerParameter& param) :BaseDataLayer < Dtype >(param) {}
	virtual void dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void loadData();
	virtual void transformData();
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	Blob<Dtype> transformed_data;
	queue<Datum*> Q;
	Batch<Dtype>* batch;
};



#endif
