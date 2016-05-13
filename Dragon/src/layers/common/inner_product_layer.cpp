#include "layers/common/inner_product_layer.hpp"

template <typename Dtype>
void InnerProductLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	InnerProductParameter inner_product_param = param.inner_product_param();
	const int num_output = inner_product_param.num_output();
	const int axis = bottom[0]->canonicalAxisIndex(inner_product_param.axis());
	bias_term = inner_product_param.bias_term();
	N = num_output;
	//	flatten channels and spatial axes into a dimension
	K = bottom[0]->count(axis);
	//	batch_size
	M = bottom[0]->count(0, axis);
	if (blobs.size()>0)	//	load previous param
		LOG(INFO) << "Checked previous params and skipped initialization";
	else{
		if (bias_term) blobs.resize(2);
		else blobs.resize(1);
		vector<int> weight_shape(2);
		//	weight as [num_output,dim]
		weight_shape[0] = N;
		weight_shape[1] = K;
		blobs[0].reset(new Blob<Dtype>(weight_shape));
		boost::shared_ptr< Filler<Dtype> > weight_filler(getFiller<Dtype>(inner_product_param.weight_filler()));
		weight_filler->fill(blobs[0].get());
		if (bias_term){
			vector<int> bias_shape(1, N);
			blobs[1].reset(new Blob<Dtype>(bias_shape));
			boost::shared_ptr< Filler<Dtype> > bias_filler(getFiller<Dtype>(inner_product_param.bias_filler()));
			bias_filler->fill(blobs[1].get());
		}
	}
	param_need_bp.resize(blobs.size(), true);
}

template<typename Dtype>
void InnerProductLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	InnerProductParameter inner_product_param = param.inner_product_param();
	const int axis = bottom[0]->canonicalAxisIndex(inner_product_param.axis());
	vector<int> top_shape = bottom[0]->shape();
	M = bottom[0]->count(0, axis);
	//	drop redundant axes
	//	we only need 2D(batch,dim) axes
	top_shape.resize(axis + 1);
	//	reset the second axis shape
	top_shape[axis] = N;
	top[0]->reshape(top_shape);
	if (bias_term){
		//	1D
		vector<int> bias_multiplier_shape(1, M);
		bias_multiplier.reshape(bias_multiplier_shape);
		dragon_set(bias_multiplier.count(), Dtype(1), bias_multiplier.mutable_cpu_data());
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype *top_data = top[0]->mutable_cpu_data();
	const Dtype* weights = blobs[0]->cpu_data();
	//	MAT[batch_size,dim] x MAT[dim,num_output]=MAT[batch_size,num_output]
	//	we replace 'Wx+b' as 'xW+b' directly
	//	it is different from conv_layer 
	//	which use for(...) to handle a batch
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
		(Dtype)1.0, bottom_data, weights, (Dtype)0.0, top_data);
	if (bias_term){
		//	mul[batch_size,1] x bias_vector[1,num_output]=bias[batch_size,num_output]
		//	top_data[batch_size,num_output] += bias[batch_size,num_output]
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, 1,
			(Dtype)1.0, bias_multiplier.cpu_data(), blobs[1]->cpu_data(), (Dtype)1.0, top_data);
	}
}

template<typename Dtype>
void InnerProductLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* weights = blobs[0]->cpu_data();
	Dtype *weights_diff = blobs[0]->mutable_cpu_diff();
	Dtype *bias_diff = blobs[1]->mutable_cpu_diff();
	Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
	if (param_need_bp[0]){

		//	weight_diff += ( bottom_data*delta_(layer+1) )
		//	use '+=' in Caffe because it will clear the diff per iter
		//	it keeps the general coding custom for layers 
		//	when handling more than one example per iter
		//	you can also replace "+=" as "="

		dragon_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N, K, M,
			(Dtype)1.0, top_diff, bottom_data, (Dtype)1.0, weights_diff);

	}
	if (bias_term && param_need_bp[1]){
		//	bias_diff += delta_(layer+1)
		//	note that gemv will choose the matrix' axis smartly
		dragon_cpu_gemv<Dtype>(CblasTrans, M, N,
			(Dtype)1.0, top_diff, bias_multiplier.cpu_data(), (Dtype)1.0, bias_diff);
	}
	if (data_need_bp[0]){
		//	bottom_diff += delta_(layer+1)*weights 
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, K, N,
			(Dtype)1.0, top_diff, weights, (Dtype)0.0, bottom_diff);
	}
}

INSTANTIATE_CLASS(InnerProductLayer);