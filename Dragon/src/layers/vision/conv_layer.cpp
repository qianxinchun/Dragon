#include "layers/vision/conv_layer.hpp"

//	we only do vaild convolution with kernel/stride/pad
template<typename Dtype>
void ConvolutionLayer<Dtype>::computeOutputShape(){
	const int* kernel_data = kernel_shape.cpu_data();
	const int* stride_data = stride.cpu_data();
	const int* pad_data = pad.cpu_data();
	output_shape.clear();
	for (int i = 0; i < num_spatial_axes; i++){
		const int input_dim = bottom_shape[channels_axis + i + 1];
		const int output_dim = (input_dim + 2 * pad_data[i] - kernel_data[i]) / stride_data[i] + 1;
		output_shape.push_back(output_dim);
	}
}

template<typename Dtype>
void DeconvolutionLayer<Dtype>::computeOutputShape(){
	const int* base_kernel = kernel_shape.cpu_data();
	const int* base_stride = stride.cpu_data();
	const int* base_pad = pad.cpu_data();
	output_shape.clear();
	for (int i = 0; i < num_spatial_axes; i++){
		const int input_dim = bottom_shape[channels_axis + i + 1];
		//	inv-op
		const int output_dim = base_stride[i] * (input_dim - 1) + base_kernel[i] - 2 * base_pad[i];
		output_shape.push_back(output_dim);
	}
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	4D(out_channels,in_channels,kernel_h,kernel_w)
	const Dtype* weights = blobs[0]->cpu_data();
	//	multi-input
	for (int i = 0; i < bottom.size(); i++){
		//	4D(batch_size,channels,height,width)
		const Dtype* bottom_data = bottom[i]->cpu_data();
		//	call reshape() to set a top blob referring to a bottom blob
		//	so top/bottom has the same blob quantity
		Dtype *top_data = top[i]->mutable_cpu_data();
		//	scan a batch
		for (int n = 0; n < num; n++){
			//	Wx
			forward_cpu_gemm(bottom_data + n*bottom_dim, weights, top_data + n*top_dim);
			if (bias_term){
				const Dtype* bias = blobs[1]->cpu_data();
				//	Wx+b
				forward_cpu_bias(top_data + n*top_dim, bias);
			}
		}
	}
}

template<typename Dtype>
void DeconvolutionLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype* weights = blobs[0]->cpu_data();
	for (int i = 0; i < bottom.size(); i++){
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype *top_data = top[i]->mutable_cpu_data();
		for (int n = 0; n < num; n++){
			backward_cpu_gemm(bottom_data + n*bottom_dim, weights, top_data + n*top_dim);
			if (bias_term){
				const Dtype* bias = blobs[1]->cpu_data();
				forward_cpu_bias(top_data + n*top_dim, bias);
			}
		}
	}
}

template<typename Dtype>
void ConvolutionLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* weights = blobs[0]->cpu_data();
	//syncedmem only allocate diff memory after we call mutable_cpu_diff()
	Dtype *weight_diff = blobs[0]->mutable_cpu_diff();
	//	multi-output
	//	we define sub-gradient as delta
	//	delta_(layer+1)=top->diff
	for (int i = 0; i < top.size(); i++){
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		if (bias_term && param_need_bp[1]){
			//cout << "CONV BACKWARD BIAS" << endl;
			Dtype *bias_diff = blobs[1]->mutable_cpu_diff();
			//	bias_diff += delta_(layer+1)
			//	a bias contributed to a channle's all spatial_dim
			//	we use gemv to combine the spatial_dim
			//	also we need sum up delta for all units in a batch
			for (int n = 0; n < num; n++)
				backward_cpu_bias(bias_diff, top_diff + n*top_dim);
		}
		if (param_need_bp[0] || data_need_bp[i]){
			//if (param_need_bp[0]) cout << "CONV BACKWARD WEIGHT" << endl;
			//if (data_need_bp[i]) cout << "CONV BACKWARD BOTTOM" << endl;
			for (int n = 0; n < num; n++){
				//	weight_diff += delta_(layer+1)*col
				//	in fully-connected layer it should be weight_diff+= delta*input
				//	we use im2col do a patch and extract the relevent input pixels in a col
				//	so in conv_layer, we replace input with col(patched input)
				//	also we need sum up delta for all units in a batch
				if (param_need_bp[0])
					weight_cpu_gemm(bottom_data + n*bottom_dim, top_diff + n*top_dim, weight_diff);
				if (data_need_bp[i])
					//	bottom_diff += delta_(layer+1)*weights
					//	bottom_diff actually is delta_(layer) and will be used in prev layer
					//	normally, bottom_diff += delta_(layer+1)*weights*f'(input)
					//	it skip the the grad of activative function
					//	we will add it in activative function layers
					backward_cpu_gemm(top_diff + n*top_dim, weights, bottom_diff + n*bottom_dim);
			}
		}
	}
}

template<typename Dtype>
void DeconvolutionLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* weights = blobs[0]->cpu_data();
	Dtype *weight_diff = blobs[0]->mutable_cpu_diff();
	for (int i = 0; i < top.size(); i++){
		const Dtype* top_diff = top[i]->cpu_diff();
		const Dtype* bottom_data = bottom[i]->cpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		if (bias_term && param_need_bp[1]){
			Dtype *bias_diff = blobs[1]->mutable_cpu_diff();
			for (int n = 0; n < num; n++)
				backward_cpu_bias(bias_diff, top_diff + n*top_dim);
		}
		if (param_need_bp[0] || data_need_bp[i]){
			for (int n = 0; n < num; n++){
				if (param_need_bp[0])
					weight_cpu_gemm(top_diff + n*top_dim, bottom_data + n*bottom_dim, weight_diff);
				if (data_need_bp[i])
					//	note that we may compute im2col for top_diff in we weight_gemm
					//	skip im2col to speed up de-conv process
					forward_cpu_gemm(top_diff + n*top_dim, weights, bottom_diff + n*bottom_dim, param_need_bp[0]);
			}
		}
	}
}

INSTANTIATE_CLASS(ConvolutionLayer);