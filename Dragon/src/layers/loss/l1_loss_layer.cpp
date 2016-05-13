#include "layers/loss/l1_loss_layer.hpp"

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//	auto add loss_weight if has not specify
	LossLayer<Dtype>::layerSetup(bottom, top);
	SmoothL1LossParameter loss_param = param.smooth_l1_loss_param();
	sigma2 = loss_param.sigma();
	//	pred_bbox_targets  
	//	gt_bbox_targets
	//  inside_weights
	//	outside_weights
	has_weights = bottom.size() >= 3;
	if (has_weights) CHECK_EQ(bottom.size(), 4);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//	reshape top=[1,1]
	LossLayer<Dtype>::reshape(bottom, top);
	//	shape=[1,36,height,width]
	if (bottom[0]->shape() != bottom[1]->shape())
		LOG(FATAL) << "SmoothL1LossLayer accpets the same shape targets.";
	if (has_weights)
		if (bottom[2]->shape() != bottom[3]->shape())
			LOG(FATAL) << "SmoothL1LossLayer accpets the same shape weights.";
	diff.reshapeLike(*bottom[0]);
	errors.reshapeLike(*bottom[0]);
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(SmoothL1LossLayer);

