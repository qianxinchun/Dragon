#include "layers/common/softmax_layer.hpp"
#include "layers/loss/softmax_loss_layer.hpp"

template <typename Dtype>
__global__ void ForwardKernel(const int n, const Dtype* prob_data, const Dtype* label_data,
	Dtype* loss_data, const int classes, const int inner_num,
	const bool has_ignore_label, const int ignore_label, Dtype* count_data){
	//	n= outer_num*inner_num
	CUDA_KERNEL_LOOP(idx, n){
		const int o_idx = idx / inner_num;
		const int i_idx = idx % inner_num;
		const int label = label_data[o_idx*inner_num + i_idx];
		if (has_ignore_label&&label == ignore_label) loss_data[idx] = count_data[idx] = 0;
		else{
			loss_data[idx] = -log(max(prob_data[(o_idx*classes + label)*inner_num + i_idx], FLT_MIN));
			count_data[idx] = 1;
		}
	}
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	// forward softmax to get prob
	softmax_layer->forward(softmax_bottom, softmax_top);
	const Dtype* prob_data = prob.gpu_data();
	const Dtype* label_data = bottom[1]->gpu_data();
	const int classes = bottom[0]->shape(axis);
	const int n = outer_num*inner_num;
	//	used as a temporary blob to store loss before merging
	//	only use the top outer_num*inner_num elements to merge
	Dtype *loss_data = bottom[0]->mutable_gpu_diff();
	//	used as a temporary blob to store the hit number
	//	only use the top outer_num*inner_num elements to merge
	Dtype *count_data = prob.mutable_gpu_diff();
	Dtype loss = 0;
	ForwardKernel<Dtype> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(
		n, prob_data, label_data, loss_data, classes, inner_num,
		has_ignore_label, ignore_label, count_data);
	//CUDA_POST_KERNEL_CHECK;
	loss = dragon_gpu_asum(n, loss_data);
	//	attention: host can not use device memory directly
	//	we must use top[0]->mutable_cpu_data() to store loss
	if (need_norm){
		int cnt = dragon_gpu_asum(n, count_data);
		top[0]->mutable_cpu_data()[0] = loss / cnt;
	}
	else top[0]->mutable_cpu_data()[0] = loss / outer_num;
	if (top.size() == 2) top[1]->shareData(prob);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BackwardKernel(const int n, const Dtype* prob_data, const Dtype* label_data,
	Dtype* bottom_diff, const int classes, const int inner_num,
	const bool has_ignore_label, const int ignore_label, Dtype* count_data){
	//	n= outer_num*inner_num
	CUDA_KERNEL_LOOP(idx, n){
		const int o_idx = idx / inner_num;
		const int i_idx = idx % inner_num;
		const int label = label_data[o_idx*inner_num + i_idx];
		if (has_ignore_label&&label == ignore_label){
			for (int c = 0; c < classes; c++)
				bottom_diff[(o_idx*classes + c)*inner_num + i_idx] = 0;
			count_data[idx] = 0;
		}
		else{
			bottom_diff[(o_idx*classes + label)*inner_num + i_idx] -= 1;
			count_data[idx] = 1;
		}
	}
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (data_need_bp[0]){
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* prob_data = prob.gpu_data();
		const Dtype* label_data = bottom[1]->gpu_data();
		Dtype *count_data = prob.mutable_gpu_diff();
		const Dtype* top_data = top[0]->gpu_data();
		int classes = bottom[0]->shape(axis);
		const int n = outer_num*inner_num;
		const int dim = prob.count() / outer_num;
		//	bottom_diff = prob_data-1 (class = label)
		//				= prob_data-0 (class ¡Ù label)
		//				= 0			  (ignore  label)
		//	see also https://www.zhihu.com/question/28927103
		dragon_gpu_copy<Dtype>(prob.count(), bottom_diff, prob_data);
		BackwardKernel<Dtype> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(
			n, prob_data, label_data, bottom_diff, classes, inner_num,
			has_ignore_label, ignore_label, count_data);
		//	usually loss_weight equal to 1 and is setted in setLossWeight()
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		//	loss/cnt => bottom_diff/cnt
		if (need_norm){
			int cnt = dragon_gpu_asum(n, count_data);
			dragon_gpu_scal<Dtype>(bottom[0]->count(), loss_weight / cnt, bottom_diff);
		}
		else dragon_gpu_scal<Dtype>(bottom[0]->count(), loss_weight / outer_num, bottom_diff);
	}
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);