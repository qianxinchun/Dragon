# ifndef BASE_DATA_LAYER_HPP
# define BASE_DATA_LAYER_HPP

#include "../../layer.hpp"
#include "../../data_transformer.hpp"
#include "../../data_reader.hpp"



template<typename Dtype>
class BaseDataLayer :public Layer<Dtype>
{
public:
	BaseDataLayer(const LayerParameter& param);
	~BaseDataLayer() {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top) = 0;
protected:
	TransformationParameter transform_param;
	boost::shared_ptr< DataTransformer<Dtype> > ptr_transformer;
	bool has_labels;
};



# endif

