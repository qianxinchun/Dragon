#include "layer_include/vision_layers.hpp"
#include "include/blob.hpp"
#include "include/filler.hpp"

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	ConvolutionParameter conv_param = param.convolution_param();
	force_nd_im2col = conv_param.force_nd_im2col();
	// default axis=1, and channel_axis=1
	channels_axis = bottom[0]->canonicalAxisIndex(conv_param.axis());
	const int first_spatial_axis = channels_axis + 1;
	const int num_axes = bottom[0]->num_axes();
	// for conv2D: 2=4-2
	num_spatial_axes = num_axes - first_spatial_axis;
	// num_spatial_axes must be greater than 0, or will do nothing
	CHECK_GT(num_spatial_axes, 0);
	vector<int> spatial_shape(1, max(num_spatial_axes, 1)); //1 x 2 
	//	set the kernel
	kernel_shape.reshape(spatial_shape);
	int *base_kernel_shape = kernel_shape.mutable_cpu_data();
	// conv2D
	if (conv_param.has_kernel_h() || conv_param.has_kernel_w()){
		CHECK_EQ(num_spatial_axes, 2) << "Kernel_w/h can only be used in Conv2D";
		CHECK_EQ(conv_param.kernel_size(), 0) << "Either kernel_size or kernel_w/h can be specfied";
		base_kernel_shape[0] = conv_param.kernel_h();
		base_kernel_shape[1] = conv_param.kernel_w();
	}else{
		const int num_kernel_dims = conv_param.kernel_size();
		//	n-dim can be repeated with a single value
		CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes);
		//	kernel_shape must be greater that 0
		for (int i = 0; i < num_spatial_axes; i++){
			base_kernel_shape[i] = conv_param.kernel(num_kernel_dims == 1 ? 0 : i);
			CHECK_GT(base_kernel_shape[i], 0) << "Kernel shape must be non-zero";
		}
	}

	//	set the stride
	stride.reshape(spatial_shape);
	int *base_stride_shape = stride.mutable_cpu_data();
	if (conv_param.has_stride_h() || conv_param.has_stride_w()){
		CHECK_EQ(num_spatial_axes, 2) << "Stride_w/h can only be used in Conv2D";
		CHECK_EQ(conv_param.stride_size(), 0) << "Either stride_size or stride_w/h can be specfied";
		base_stride_shape[0] = conv_param.stride_h();
		base_stride_shape[1] = conv_param.stride_w();
	}else{
		const int num_stride_dims = conv_param.stride_size();
		CHECK(num_stride_dims == 0 || num_stride_dims == 1 || num_stride_dims == num_spatial_axes);
		const int defaultStride = 1;
		for (int i = 0; i < num_spatial_axes; i++){
			if (num_stride_dims == 0) base_stride_shape[i] = defaultStride;
			else base_stride_shape[i] = conv_param.stride(num_stride_dims == 1 ? 0 : i);
			CHECK_GT(base_stride_shape[i], 0) << "Stride shape must be non-zero";
		}
	}

	//	set the pad
	pad.reshape(spatial_shape);
	int *base_pad_shape = pad.mutable_cpu_data();
	if (conv_param.has_pad_h() || conv_param.has_pad_w()){
		CHECK_EQ(num_spatial_axes, 2) << "Pad_w/h can only be used in Conv2D";
		CHECK_EQ(conv_param.stride_size(), 0) << "Either pad_size or stride_w/h can be specfied";
		base_stride_shape[0] = conv_param.pad_h();
		base_stride_shape[1] = conv_param.pad_w();
	}
	else{
		const int num_pad_dims = conv_param.pad_size();
		CHECK(num_pad_dims == 0 || num_pad_dims == 1 || num_pad_dims == num_spatial_axes);
		const int defaultStride = 0;
		for (int i = 0; i < num_spatial_axes; i++){
			if (num_pad_dims == 0) base_pad_shape[i] = defaultStride;
			else base_pad_shape[i] = conv_param.pad(num_pad_dims == 1 ? 0 : i);
		}
	}

	//check the 1x1 case
	is_1x1 = true;
	for (int i = 0; i < num_spatial_axes; i++){
		is_1x1 &= (base_kernel_shape[i] == 1 && base_stride_shape[i] == 1 && base_pad_shape[i] == 0);
		if (!is_1x1) break;
	}


	channels = bottom[0]->shape(channels_axis);
	num_output = conv_param.num_output();
	CHECK_GT(num_output, 0);
	group = conv_param.group();
	CHECK_EQ(channels % group, 0);
	CHECK_EQ(num_output % group, 0);
	if (reverseDimensions()){
		conv_out_channels = channels;
		conv_in_channels = num_output;
	}else{
		conv_out_channels = num_output;
		conv_in_channels = channels;
	}
	vector<int> weight_shape(2);
	//	4D (out,in,kernel_h,kernel_w)
	weight_shape[0] = conv_out_channels;
	//	group make some feature maps overlap
	weight_shape[1] = conv_in_channels / group;
	for (int i = 0; i < num_spatial_axes; i++)
		weight_shape.push_back(base_kernel_shape[i]);
	bias_term = conv_param.bias_term();
	//  1D (num_output)
	vector<int> bias_shape(bias_term, num_output);
	if (blobs.size()>0){   //	load previous params
		CHECK_EQ(bias_term + 1, blobs.size()) << "Loading wrong number of the weight/bias blob";
		if (weight_shape != blobs[0]->shape())
			LOG(FATAL) << "Loading wrong shape of the weight shape";
		if (bias_term&&bias_shape != blobs[1]->shape())
			LOG(FATAL) << "Loading wrong shape of the bias shape";
		LOG(INFO) << "Checked previous params and skipped initialization";
	}else{	//	initialization
		if (bias_term) blobs.resize(2);
		else blobs.resize(1);
		blobs[0].reset(new Blob<Dtype>(weight_shape));
		boost::shared_ptr< Filler<Dtype> > weight_filler(getFiller<Dtype>(conv_param.weight_filler()));
		weight_filler->fill(blobs[0].get());
		if (bias_term){
			blobs[1].reset(new Blob<Dtype>(bias_shape));
			boost::shared_ptr< Filler<Dtype> > bias_filler(getFiller<Dtype>(conv_param.bias_filler()));
			bias_filler->fill(blobs[1].get());
		}
	}
	//	channels_in/group * kernel_h * kernel_w
	kernel_dim = blobs[0]->count(1);
	//	channels_out*channels_in*kernel_h*kernel_w / group
	weight_offset = conv_out_channels * kernel_dim / group;
	// you can specfic weight/bias whether to participate back-propagation
	param_need_bp.resize(blobs.size(), true);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const int first_spatial_axis = channels_axis + 1;
	CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes)
		<< "bottom axes can not change";
	num = bottom[0]->count(0, channels_axis);
	//	labels do not participate forward propogation
	//	see also prototxt file
	for (int bottom_id=1; bottom_id < bottom.size(); bottom_id++){
		CHECK(bottom[bottom_id]->shape() == bottom[0]->shape()) 
			<< "All layers in a branch must have the same input shape";
	}
	bottom_shape = bottom[0]->shape();
	//	compute the output_shape for top_shape using kernel/stride/pad like im2col
	//	it is a pure virtual function and will implement in ConvlutionLayer
	computeOutputShape();
	//	stuff batch_size into top_shape
	//	Blob.shape() must be const vector<int>&
	//	or will get wrong iterator
	vector<int> top_shape(bottom[0]->shape().begin(), bottom[0]->shape().begin() + channels_axis);
	//	stuff output_maps into top_shape
	top_shape.push_back(num_output);
	//	stuff map_shape into top_shape
	for (int i = 0; i < num_spatial_axes; i++) top_shape.push_back(output_shape[i]);
	//	reshape all top blobs(batch,num_output,out_height,out,width)
	for (int top_id = 0; top_id < top.size(); top_id++) top[top_id]->reshape(top_shape);
	if (reverseDimensions()) conv_out_spatial_dim = bottom [0]->count(first_spatial_axis);
	else conv_out_spatial_dim = top[0]->count(first_spatial_axis);
	//	5D result
	col_offset = kernel_dim*conv_out_spatial_dim;
	//	3D result, why divide group at the output?
	output_offset = conv_out_channels*conv_out_spatial_dim / group;
	//	3D(channels,height,width)
	vector<int> bottom_dim_blob_shape(1, num_spatial_axes + 1);
	conv_input_shape.reshape(bottom_dim_blob_shape);
	int *base_conv_input_shape = conv_input_shape.mutable_cpu_data();
	for (int i = 0; i < num_spatial_axes + 1; i++){
		if (reverseDimensions())	base_conv_input_shape[i] = top[0]->shape(channels_axis + i);
		else base_conv_input_shape[i] = bottom[0]->shape(channels_axis + i);
	}
	//	3D
	col_buffer_shape.clear();
	//kernel_dim is divided by group, but buffer block need full size with group
	col_buffer_shape.push_back(kernel_dim*group);
	for (int i = 0; i < num_spatial_axes; i++){
		if (reverseDimensions()) col_buffer_shape.push_back(bottom_shape[channels_axis + i+1]);
		else col_buffer_shape.push_back(output_shape[i]);
	}
	col_buffer.reshape(col_buffer_shape);
	bottom_dim = bottom[0]->count(channels_axis);
	top_dim = top[0]->count(channels_axis);
	// 3D result (channel*blob_height*blob_width)
	num_kernels_im2col = conv_in_channels*conv_out_spatial_dim;
	num_kernels_col2im = reverseDimensions() ? top_dim : bottom_dim;
	out_spatial_dim = top[0]->count(first_spatial_axis);
	//	a temp vector to make bias vector into a bias mat
	//	we get like this:	bias_vector[num_out,1] x mul[1,out_spatial_dim]
	//	=	bias[num_out,out_spatial_dim]
	//	pay attention that bias will not be affected by reverseDimensions()
	//	it use num_out and out_spatial_dim
	//	but not conv_out and conv_out_spatial_dim
	if (bias_term){
		//	1D
		vector<int> bias_multiplier_shape(1, out_spatial_dim);
		bias_multiplier.reshape(bias_multiplier_shape);
		dragon_set(bias_multiplier.count(), Dtype(1.0), bias_multiplier.mutable_cpu_data());
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col){
	const Dtype* col_buff_ = input;
	//	1x1 is a special case, im2col do nothing so we needn't do it to waste time
	if (!is_1x1){
		if (!skip_im2col) conv_im2col_cpu(input, col_buffer.mutable_cpu_data());
		// after patch just use const data
		col_buff_ = col_buffer.cpu_data();
	}
	for (int g = 0; g < group; g++){
		//	MAT[out_channels,kernel_dim] x MAT[kernel_dim,conv_out_spatial_dim]
		//	=MAT[out_channels,conv_out_spatial_dim]
		//	kernel_dim using input_channels/group in Matrix-Product
		//	so each output Mat has the sum of subset channles directly
		//	weight_off and output_off decide the output map in different groups
		//	col_off decide the input map in differnet groups
		//	group convolution is the method in LeNet5 or early convnet
		//	actually it is almost useless but also implemented in Caffe Framework
		//	refer to https://www.zhihu.com/question/26871787/answer/38935261 @Yangqing Jia
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels / group,
			conv_out_spatial_dim, kernel_dim, (Dtype)1.0, weights + weight_offset*g, col_buff_ + col_offset*g,
			(Dtype)0.0, output + output_offset*g); 
	}
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output, const Dtype* bias){
	//	bias_vector[num_out,1] x mul[1,out_spatial_dim]=bias[num_out,out_spatial_dim]
	//	bias_multiplier see also at the end of reshape()
	//	from this gemm op we get a bias matrix
	//	then we could do mat_weight + mat_bias use the param_beta in cpu_gemm
	//	final_out=alpha*bias_vec*bias_mul+beta*weight_out (alpha=beta=1.0)
	//	final_out matrix should be splitted into a 3D(channels,height,width) form then
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output, out_spatial_dim, 1, (Dtype)1.0,
		bias, bias_multiplier.cpu_data(), (Dtype)1.0, output);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input){
	Dtype* col_buff_ = col_buffer.mutable_cpu_data();
	// we do not copy the data into col_buff at the condition of 1x1
	if (is_1x1) col_buff_ = input;
	//	MAT[kernel_dim,out_channels] x MAT[out_channels,conv_out_spatial_dim]
	//	=MAT[kernel_dim,conv_out_spatial_dim]
	//	it is a inverse op comparing to forward_cpu_gemm()
	//	W*C => O, W^(T)*O => W^(T)W*C 
	//	we use weight and output to get col
	//	transpose op in blas must specific lda/ldb/ldc explicitly using CblasTrans flag in gemm op
	//	more see dragon_cpu_gemm() in math_functions.cpp
	for (int g = 0; g < group; g++){
		dragon_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim, conv_out_spatial_dim, conv_out_channels / group,
			(Dtype)1.0, weights + weight_offset*g, output + output_offset*g, (Dtype)0.0, col_buff_ + col_offset*g);
	}
	//	1x1 also do nothing
	if (!is_1x1){
		conv_col2im_cpu(col_buff_, input);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype *weights){
	const Dtype *col_buff_ = input;
	//	patch the input as col_buff before
	if (!is_1x1){
		conv_im2col_cpu(input, col_buffer.mutable_cpu_data());
		col_buff_ = col_buffer.cpu_data();
	}
	for (int g = 0; g < group; g++){
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels / group,
			kernel_dim, conv_out_spatial_dim, (Dtype)1.0, output + output_offset*g,
			col_buff_ + col_offset*g, (Dtype)1.0, weights + weight_offset*g);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias, const Dtype* input){
	dragon_cpu_gemv<Dtype>(CblasNoTrans, num_output, out_spatial_dim, (Dtype)1.0,
		input, bias_multiplier.cpu_data(), (Dtype)1.0,bias);
}


#ifndef CPU_ONLY
template<typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col){
	const Dtype* col_buff_ = input;
	if (!is_1x1){
		if (!skip_im2col) conv_im2col_gpu(input, col_buffer.mutable_gpu_data());
		col_buff_ = col_buffer.gpu_data();
	}
	for (int g = 0; g < group; g++){
		dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels / group,
			conv_out_spatial_dim, kernel_dim, (Dtype)1.0, weights + weight_offset*g, col_buff_ + col_offset*g,
			(Dtype)0.0, output + output_offset*g);
	}
}
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output, const Dtype* bias){
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output, out_spatial_dim, 1, (Dtype)1.0,
		bias, bias_multiplier.gpu_data(), (Dtype)1.0, output);
}
template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input){
	Dtype* col_buff_ = col_buffer.mutable_gpu_data();
	if (is_1x1) col_buff_ = input;
	for (int g = 0; g < group; g++){
		dragon_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim, conv_out_spatial_dim, conv_out_channels / group,
			(Dtype)1.0, weights + weight_offset*g, output + output_offset*g, (Dtype)0.0, col_buff_ + col_offset*g);
	}
	if (!is_1x1) conv_col2im_gpu(col_buff_, input);
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype *weights){
	const Dtype *col_buff_ = input;
	if (!is_1x1){
		conv_im2col_gpu(input, col_buffer.mutable_gpu_data());
		col_buff_ = col_buffer.gpu_data();
	}
	for (int g = 0; g < group; g++){
		dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels / group,
			kernel_dim, conv_out_spatial_dim, (Dtype)1.0, output + output_offset*g,
			col_buff_ + col_offset*g, (Dtype)1.0, weights + weight_offset*g);
	}
}

template<typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias, const Dtype* input){
	dragon_gpu_gemv<Dtype>(CblasNoTrans, num_output, out_spatial_dim, (Dtype)1.0,
		input, bias_multiplier.gpu_data(), (Dtype)1.0, bias);
}
#endif

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
		if (param_need_bp[0]||data_need_bp[i]){
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

INSTANTIATE_CLASS(BaseConvolutionLayer);
INSTANTIATE_CLASS(ConvolutionLayer);