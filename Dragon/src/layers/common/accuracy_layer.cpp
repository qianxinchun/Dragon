#include "layers/common/accuracy_layer.hpp"

template <typename Dtype>
void AccuracyLayer < Dtype >::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	default top_k=1 and select the max prob
	top_k = param.accuracy_param().top_k();
	has_ignore_label = param.accuracy_param().has_ignore_label();
	if (has_ignore_label) ignore_label = param.accuracy_param().ignore_label();
}

template <typename Dtype>
void AccuracyLayer < Dtype >::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	CHECK_LE(top_k, bottom[0]->count() / bottom[1]->count())
		<< "Top_k must less equal than total classes.";
	axis = bottom[0]->canonicalAxisIndex(param.accuracy_param().axis());
	outer_num = bottom[0]->count(0, axis);
	inner_num = bottom[0]->count(axis + 1);
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
				//cout << "prob: "<<vec[k].first<< "    pred: " << vec[k].second << "   label: " << label << endl;
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

INSTANTIATE_CLASS(AccuracyLayer);