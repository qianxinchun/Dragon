#include "layer_include/loss_layers.hpp"
#include "layer_include/vision_layers.hpp"
#include "layer_include/layer_factory.hpp"
#include "algorithm"
template <typename Dtype>
void LossLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	// auto add solve loss option if it is a LossLayer
	if (param.loss_weight_size() == 0) param.add_loss_weight(Dtype(1));
}

template <typename Dtype>
void LossLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
		<< "Data and Label should have same size.";
	//	loss just need an memory unit for storing
	vector<int> loss_shape(1, 1);
	top[0]->reshape(loss_shape);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	LossLayer<Dtype>::layerSetup(bottom, top);
	LayerParameter softmax_param(param);
	softmax_param.set_type("Softmax");
	softmax_layer.reset(new SoftmaxLayer<Dtype>(param));
	//	data
	softmax_bottom.clear();
	softmax_bottom.push_back(bottom[0]);
	//	output prob 
	softmax_top.clear();
	softmax_top.push_back(&prob);
	//	we only use (data,prob) for SoftmaxLayer
	softmax_layer->setup(softmax_bottom,softmax_top);
	has_ignore_label = param.loss_param().has_ignore_label();
	has_normalize = param.loss_param().has_normalize();
	if (has_ignore_label) ignore_label = param.loss_param().ignore_label();
	if (has_normalize) need_norm = param.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	SoftmaxParameter softmax_param = param.softmax_param();
	LossLayer<Dtype>::reshape(bottom, top);
	softmax_layer->reshape(softmax_bottom, softmax_top);
	//	we regard the axis has the classes we need to classify
	//	and other axes will be merged as the number of examples
	//	and you need not use 3D/4D input
	//	use a 2D inner product layer before Softmax is the best choice
	axis = bottom [0]->canonicalAxisIndex(param.softmax_param().axis());
	//	left part of the class num axis
	outer_num = bottom[0]->count(0, axis);
	//	right part of the class nun axis
	inner_num = bottom[0]->count(axis + 1);
	CHECK_EQ(outer_num*inner_num, bottom[1]->count())
		<< "Number of predictions must match the number of labels.";
	//	original softmax prob output if need
	if (top.size() >= 2) top[1]->reshapeLike(*bottom[0]);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	// forward softmax to get prob
	softmax_layer->forward(softmax_bottom, softmax_top); 
	const Dtype* prob_data = prob.cpu_data();
	const Dtype* label_data = bottom[1]->cpu_data();
	Dtype *top_data = top[0]->mutable_cpu_data();
	int dim = prob.count() / outer_num;
	int cnt = 0;
	Dtype loss = 0;
	//	for each example
	for (int i = 0; i < outer_num; i++){
		//	for each label in a example
		for (int j = 0; j < inner_num; j++){
			const int label = label_data[i*inner_num + j];
			if (has_ignore_label&&label == ignore_label) continue;
			//	start from zero
			CHECK_GE(label, 0); 
			//	max value must less than max classes
			CHECK_LT(label, prob.shape(axis));
			//	we can regrad prob data as a 2D matrix for each example
			//	MAT[classes,inner] and label*inner_num+j will get the j_th inner's label prob
			Dtype labeled_prob = prob_data[i*dim + label*inner_num + j];
			//	FLT_MIN=1.175494351e-38F
			//	re-tune the output prob in case the numerical issue due to bug in log
			//	log(0) is terrible but log(FLT_MIN) not
			labeled_prob = max<Dtype>(labeled_prob, FLT_MIN);
			//	sum up negative log loss
			loss -= log(labeled_prob);
			cnt++;
		}
	}
	//	average all labels 
	if (need_norm) top_data[0] = loss / cnt;
	else top_data[0] = loss / outer_num;
	if (top.size() ==2) top[1]->shareData(prob);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, 
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (data_need_bp[1]) LOG(FATAL) << "Labels can not do back propogation.";
	if (data_need_bp[0]){
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* prob_data = prob.cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		//	bottom_diff = prob_data-1 (class = label)
		//				= prob_data-0 (class ¡Ù label)
		//				= 0			  (ignore  label)
		//	see also https://www.zhihu.com/question/28927103
		dragon_copy<Dtype>(prob.count(), bottom_diff, prob_data);
		int dim = prob.count() / outer_num;
		int cnt = 0;
		for (int i = 0; i < outer_num; i++){
			for (int j = 0; j < inner_num; j++){
				const int label = label_data[i*inner_num + j];
				if (has_ignore_label&&label == ignore_label){
					//	if we want to kill a label's gradient
					//	we must clear for all classses(both [prob_data-1] and [prob_data])
					for (int c = 0; c < bottom[0]->shape(axis); c++) bottom_diff[i*dim + c*inner_num + j] = 0;
				}else{
					bottom_diff[i*dim + label*inner_num + j] -= 1;
					cnt++;
				}
			}
		}
		//	usually loss_weight equal to 1 and is setted in setLossWeight()
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		//	loss/cnt => bottom_diff/cnt
		if (need_norm) dragon_scal<Dtype>(bottom[0]->count(), loss_weight / cnt, bottom_diff);
		else dragon_scal<Dtype>(bottom[0]->count(), loss_weight / outer_num, bottom_diff);
	}
}

template <typename Dtype>
void AccuracyLayer < Dtype >::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	default top_k=1 and select the max prob
	top_k = param.accuracy_param().top_k();
	has_ignore_label = param.accuracy_param().has_ignore_label();
	if (has_ignore_label) ignore_label=param.accuracy_param().ignore_label();
}

template <typename Dtype>
void AccuracyLayer < Dtype >::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	CHECK_LE(top_k, bottom[0]->count() / bottom[1]->count())
		<< "Top_k must less equal than total classes.";
	axis = bottom[0]->canonicalAxisIndex(param.accuracy_param().axis());
	outer_num = bottom[0]->count(0, axis);
	inner_num = bottom[0]->count(axis+1);
	CHECK_EQ(outer_num*inner_num, bottom[1]->count());
	vector<int> top_shape(1, 1);
	top[0]->reshape(top_shape);
	if (top.size() > 1){
		vector<int> top_shape_per_class(1);
		//	 total number of classes
		top_shape_per_class[0] = bottom[0]->shape(axis);
		top[1]->reshape(top_shape_per_class);
		nums_buffer.reshape(top_shape_per_class);
	}
}

template <typename Dtype>
void AccuracyLayer < Dtype >::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	Dtype accuracy = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label_data = bottom[1]->cpu_data();
	const int dim = bottom[0]->count() / outer_num;
	const int num_labels = bottom[0]->shape(axis);
	vector<Dtype> max_val(top_k + 1);
	vector<Dtype> max_id(top_k + 1);
	if (top.size() > 1){
		dragon_set(nums_buffer.count(), Dtype(0), nums_buffer.mutable_cpu_data()); 
		dragon_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
	}
	int count = 0;
	for (int i = 0; i < outer_num; i++){
		for (int j = 0; j < inner_num; j++){
			const int label = label_data[i*inner_num + j];
			if (has_ignore_label&&label == ignore_label) continue;
			//	statistics
			if (top.size()>1) nums_buffer.mutable_cpu_data()[label]++;
			CHECK_GE(label, 0);
			//	max value must less than max classes
			CHECK_LT(label, num_labels);
			//	fill into the values for comparsion from different classes
			vector<pair<Dtype, int>> vec;
			for (int k = 0; k < num_labels; k++)
				vec.push_back(make_pair(bottom_data[i*dim + k*inner_num + j], k));
			//	compare pair.first then pair.second
			//	only execute for top_k values
			partial_sort(vec.begin(), vec.begin() + top_k, vec.end(), greater<pair<Dtype, int>>());
			for (int k = 0; k < top_k; k++){
				//	hit
				if (vec[k].second == label){
					accuracy++;
					//	hit statistics
					if (top.size()>1) top[1]->mutable_cpu_data()[label]++;
					break;
				}
			}
			//	objects
			count++;
		}	//	end inner_num
	}	// end outer_num
	top[0]->mutable_cpu_data()[0] = accuracy / count;
	if (top.size()>1){
		//	statistics for each classes
		for (int i = 0; i < top[1]->count(); i++)
			top[1]->mutable_cpu_data()[i] =
			nums_buffer.cpu_data()[i] == 0 ? 0 : top[1]->cpu_data()[i] / nums_buffer.cpu_data()[i];
	}
}




INSTANTIATE_CLASS(LossLayer);
INSTANTIATE_CLASS(SoftmaxWithLossLayer);
INSTANTIATE_CLASS(AccuracyLayer);