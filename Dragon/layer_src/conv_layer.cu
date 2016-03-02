#include "layer_include/vision_layers.hpp"

template<typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	4D(out_channels,in_channels,kernel_h,kernel_w)
	const Dtype* weights = blobs[0]->gpu_data();
	//	multi-input
	for (int i = 0; i < bottom.size(); i++){
		//	4D(batch_size,channels,height,width)
		const Dtype* bottom_data = bottom[i]->gpu_data();
		//	call reshape() to set a top blob referring to a bottom blob
		//	so top/bottom has the same blob quantity
		Dtype *top_data = top[i]->mutable_gpu_data();
		//	scan a batch
		for (int n = 0; n < num; n++){
			//	Wx
			forward_gpu_gemm(bottom_data + n*bottom_dim, weights, top_data + n*top_dim);
			if (bias_term){
				const Dtype* bias = blobs[1]->gpu_data();
				//	Wx+b
				forward_gpu_bias(top_data + n*top_dim, bias);
			}
		}
	}
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* weights = blobs[0]->gpu_data();
	//syncedmem only allocate diff memory after we call mutable_cpu_diff()
	Dtype *weight_diff = blobs[0]->mutable_gpu_diff();
	//	multi-output
	//	we define sub-gradient as delta
	//	delta_(layer+1)=top->diff
	for (int i = 0; i < top.size(); i++){
		const Dtype* top_diff = top[i]->gpu_diff();
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		if (bias_term && param_need_bp[1]){
			Dtype *bias_diff = blobs[1]->mutable_gpu_diff();
			//	bias_diff += delta_(layer+1)
			//	a bias contributed to a channle's all spatial_dim
			//	we use gemv to combine the spatial_dim
			//	also we need sum up delta for all units in a batch
			for (int n = 0; n < num; n++)
				backward_gpu_bias(bias_diff, top_diff + n*top_dim);
		}
		if (param_need_bp[0] || data_need_bp[i]){
			for (int n = 0; n < num; n++){
				//	weight_diff += delta_(layer+1)*col
				//	in fully-connected layer it should be weight_diff+= delta*input
				//	we use im2col do a patch and extract the relevent input pixels in a col
				//	so in conv_layer, we replace input with col(patched input)
				//	also we need sum up delta for all units in a batch
				if (param_need_bp[0])
					weight_gpu_gemm(bottom_data + n*bottom_dim, top_diff + n*top_dim, weight_diff);
				if (data_need_bp[i])
					//	bottom_diff += delta_(layer+1)*weights
					//	bottom_diff actually is delta_(layer) and will be used in prev layer
					//	normally, bottom_diff += delta_(layer+1)*weights*f'(input)
					//	it skip the the grad of activative function
					//	we will add it in activative function layers
					backward_gpu_gemm(top_diff + n*top_dim, weights, bottom_diff + n*bottom_dim);
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);