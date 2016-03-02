#include "layer_include/vision_layers.hpp"
#include "include/math_function.hpp"

template<typename Dtype>
void InnerProductLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype *top_data = top[0]->mutable_gpu_data();
	const Dtype* weights = blobs[0]->gpu_data();
	//	MAT[batch_size,dim] x MAT[dim,num_output]=MAT[batch_size,num_output]
	//	we replace 'Wx+b' as 'xW+b' directly
	//	it is different from conv_layer 
	//	which use for(...) to handle a batch
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
		(Dtype)1.0, bottom_data, weights, (Dtype)0.0, top_data);
	if (bias_term){
		//	mul[batch_size,1] x bias_vector[1,num_output]=bias[batch_size,num_output]
		//	top_data[batch_size,num_output] += bias[batch_size,num_output]
		dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1,
			(Dtype)1.0, bias_multiplier.gpu_data(), blobs[1]->gpu_data(), (Dtype)1.0, top_data);
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* weights = blobs[0]->gpu_data();
	Dtype *weights_diff = blobs[0]->mutable_gpu_diff();
	Dtype *bias_diff = blobs[1]->mutable_gpu_diff();
	Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
	if (param_need_bp[0]){

		//	weight_diff += ( bottom_data*delta_(layer+1) )
		//	use '+=' in Caffe because it will clear the diff per iter
		//	it keeps the general coding custom for layers 
		//	when handling more than one example per iter
		//	you can also replace "+=" as "="

		/* Free Style */
		/*dragon_cpu_gemm(CblasTrans, CblasNoTrans, K, N, M,
		(Dtype)1.0, bottom_data, top_diff, (Dtype)0.0, weights_diff);*/

		/* General Coding Custom Style */
		dragon_gpu_gemm(CblasTrans, CblasNoTrans, K, N, M,
			(Dtype)1.0, bottom_data, top_diff, (Dtype)1.0, weights_diff);

	}
	if (bias_term && param_need_bp[1]){
		//	bias_diff += delta_(layer+1)
		//	note that gemv will choose the last axis smartly
		dragon_gpu_gemv(CblasTrans, M, N,
			(Dtype)1.0, top_diff, bias_multiplier.gpu_data(), (Dtype)1.0, bias_diff);
	}
	if (data_need_bp[0]){
		//	bottom_diff += delta_(layer+1)*weights 
		dragon_gpu_gemm(CblasNoTrans, CblasTrans, M, K, N,
			(Dtype)1.0, top_diff, weights, (Dtype)0, bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);