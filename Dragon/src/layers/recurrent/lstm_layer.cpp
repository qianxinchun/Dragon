#include "layers/recurrent/lstm_layer.hpp"

#define INPUT_GATE d
#define FORGET_GATE hidden_dim+d
#define OUTPUT_GATE 2*hidden_dim+d
#define CELL_GATE 3*hidden_dim+d
template <typename Dtype>
Dtype sigmoid(Dtype x){
	return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
void LSTMLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	LSTMParameter lstm_param = param.lstm_param();
	clipping_threshold = lstm_param.clipping_threshold();
	batch_size = lstm_param.batch_size();
	hidden_dim = lstm_param.num_output();
	input_dim = bottom[0]->count() / bottom[0]->num();
	if (blobs.size() > 0){
		LOG(INFO) << "Checked previous params and skipped initialization";
	}
	else{
		blobs.resize(3);
		boost::shared_ptr<Filler<Dtype>> weight_filler
			(getFiller<Dtype>(lstm_param.weight_filler()));
		vector<int> weight_shape;
		//	use weight concatenation
		//	see more in http://www.deeplearning.net/tutorial/lstm.html
		//	weight for input
		weight_shape.push_back(4 * hidden_dim);
		weight_shape.push_back(input_dim);
		blobs[0].reset(new Blob<Dtype>(weight_shape));
		weight_filler->fill(blobs[0].get());

		//	weight for hidden units
		weight_shape.clear();
		weight_shape.push_back(4 * hidden_dim);
		weight_shape.push_back(hidden_dim);
		blobs[1].reset(new Blob<Dtype>(weight_shape));
		weight_filler->fill(blobs[1].get());

		// bias
		weight_shape.clear();
		weight_shape.resize(1, 4 * hidden_dim);
		blobs[2].reset(new Blob<Dtype>(weight_shape));
		boost::shared_ptr<Filler<Dtype>> bias_filler
			(getFiller<Dtype>(lstm_param.bias_filler()));
		bias_filler->fill(blobs[2].get());
	}
	param_need_bp.resize(blobs.size(), true);

	//	cell_shape is [batch_size,hidden_dim]
	vector<int> cell_shape;
	cell_shape.push_back(batch_size);
	cell_shape.push_back(hidden_dim);
	c_1.reshape(cell_shape);
	c_T.reshape(cell_shape);
	h_1.reshape(cell_shape);
	h_T.reshape(cell_shape);
	h_to_h.reshape(cell_shape);

	//	4 gates use same the memory blob
	vector<int> gate_shape;
	gate_shape.push_back(batch_size);
	gate_shape.push_back(4);
	gate_shape.push_back(hidden_dim);
	h_to_gate.reshape(gate_shape);
}

template <typename Dtype>
void LSTMLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	num represents sentence_lenth*batch_size
	//	channels represent input_dim
	steps = bottom[0]->num() / batch_size;
	CHECK_EQ(bottom[0]->num() % batch_size, 0);
	CHECK_EQ(bottom[0]->count() / batch_size / steps, input_dim);
	vector<int> top_shape;
	top_shape.push_back(batch_size*steps);
	top_shape.push_back(hidden_dim);
	top[0]->reshape(top_shape);

	//	use inverse(steps|batch_size) matrix
	vector<int> gate_shape;
	gate_shape.push_back(steps);
	gate_shape.push_back(batch_size);
	gate_shape.push_back(4);
	gate_shape.push_back(hidden_dim);
	pre_gate.reshape(gate_shape);
	gate.reshape(gate_shape);

	vector<int> output_shape;
	output_shape.push_back(steps);
	output_shape.push_back(batch_size);
	output_shape.push_back(hidden_dim);
	cell.reshape(output_shape);
	output.reshape(output_shape);
	//	same memory but different axes
	output.shareData(*top[0]);
	output.shareDiff(*top[0]);

	vector<int> multiplier_shape(1, batch_size*steps);
	bias_multiplier.reshape(multiplier_shape);
	dragon_set(bias_multiplier.count(), Dtype(1), bias_multiplier.mutable_cpu_data());
}

template <typename Dtype>
void LSTMLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	CHECK_EQ(top[0]->cpu_data(), output.cpu_data());
	Dtype* top_data = output.mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* clip = NULL;
	//	1D
	if (bottom.size() > 1){
		clip = bottom[1]->cpu_data();
		CHECK_EQ(bottom[1]->num(), bottom[1]->count());
	}
	const Dtype* W = blobs[0]->cpu_data();
	const Dtype* U = blobs[1]->cpu_data();
	const Dtype* b = blobs[2]->cpu_data();

	Dtype* pre_gate_data = pre_gate.mutable_cpu_data();
	Dtype* gate_data = gate.mutable_cpu_data();
	Dtype* cell_data = cell.mutable_cpu_data();
	Dtype* h_to_gate_data = h_to_gate.mutable_cpu_data();

	if (clip){
		dragon_copy<Dtype>(c_1.count(), c_1.mutable_cpu_data(), c_T.cpu_data());
		dragon_copy<Dtype>(h_1.count(), h_1.mutable_cpu_data(), h_T.cpu_data());
	}
	else{
		//	c(-1) and h(-1) should be set to zero
		dragon_set<Dtype>(c_1.count(), Dtype(0), c_1.mutable_cpu_data());
		dragon_set<Dtype>(h_1.count(), Dtype(0), h_1.mutable_cpu_data());
	}

	//	comupte Wx for all gates
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, steps*batch_size, 4 * hidden_dim, input_dim,
		Dtype(1), bottom_data, W, Dtype(0), pre_gate_data);
	//	compute Wx+b for all gates
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, steps*batch_size, 4 * hidden_dim, 1,
		Dtype(1), bias_multiplier.cpu_data(), b, Dtype(1), pre_gate_data);
	//	scan for all steps
	for (int t = 0; t < steps; t++){
		Dtype* h_t = top_data + output.offset(t);
		Dtype* c_t = cell_data + cell.offset(t);
		Dtype* pre_gate_t = pre_gate_data + pre_gate.offset(t);
		Dtype* gate_t = gate_data + gate.offset(t);
		const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
		//	use h(-1) and c(-1) when t=0
		const Dtype* h_t_1 = t > 0 ? (h_t - output.offset(1)) : h_1.cpu_data();
		const Dtype* c_t_1 = t > 0 ? (c_t - cell.offset(1)) : c_1.cpu_data();

		//	compute U*h(t-1) in h_to_gate
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, batch_size, 4 * hidden_dim, hidden_dim,
			Dtype(1), h_t_1, U, Dtype(0), h_to_gate_data);

		for (int n = 0; n < batch_size; n++){
			bool cont = clip_t ? clip_t[n]>0 : t > 0;
			//	apply U*h(t-1) when t>0
			if (cont) dragon_add<Dtype>(4 * hidden_dim, pre_gate_t, h_to_gate_data, pre_gate_t);

			for (int d = 0; d < hidden_dim; d++){
				//	sigmoid for gates
				gate_t[INPUT_GATE] = sigmoid(pre_gate_t[INPUT_GATE]);
				//	forget_gate only can be used when t>0
				gate_t[FORGET_GATE] = cont ? sigmoid(pre_gate_t[FORGET_GATE]) : Dtype(0);
				gate_t[OUTPUT_GATE] = sigmoid(pre_gate_t[OUTPUT_GATE]);
				gate_t[CELL_GATE] = tanh(pre_gate_t[CELL_GATE]);
				//	c(t)=i(t)*g(t)+f(t)*c(t-1)
				c_t[d] = gate_t[INPUT_GATE] * gate_t[CELL_GATE] + gate_t[FORGET_GATE] * c_t_1[d];
				//	h(t)=o(t)*tanh(c(t))
				h_t[d] = gate_t[OUTPUT_GATE] * tanh(c_t[d]);
			}
			h_t += hidden_dim;
			c_t += hidden_dim;
			c_t_1 += hidden_dim;
			pre_gate_t += 4 * hidden_dim;
			gate_t += 4 * hidden_dim;
			h_to_gate_data += 4 * hidden_dim;
		}
	}	//end steps

	//	store T-1 in T for BPTT
	//	it seems useless in https://github.com/junhyukoh/caffe-lstm/blob/master/src/caffe/layers/lstm_layer.cpp
	dragon_copy<Dtype>(batch_size*hidden_dim, c_T.mutable_cpu_data(),
		cell_data + cell.offset(steps - 1));
	dragon_copy<Dtype>(batch_size*hidden_dim, h_T.mutable_cpu_data(),
		top_data + output.offset(steps - 1));
}

template <typename Dtype>
void LSTMLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* top_data = output.cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* clip = NULL;
	if (bottom.size() > 1){
		clip = bottom[1]->cpu_data();
		CHECK_EQ(bottom[1]->num(), bottom[1]->count());
	}
	const Dtype* W = blobs[0]->cpu_data();
	const Dtype* U = blobs[1]->cpu_data();
	const Dtype* gate_data = gate.cpu_data();
	const Dtype* cell_data = cell.cpu_data();

	Dtype* top_diff = output.mutable_cpu_diff();
	Dtype* pre_gate_diff = pre_gate.mutable_cpu_diff();
	Dtype* gate_diff = gate.mutable_cpu_diff();
	Dtype* cell_diff = cell.mutable_cpu_diff();

	//	only copy zero actually
	dragon_copy<Dtype>(batch_size*hidden_dim, cell_diff + cell.offset(steps - 1),
		c_T.cpu_diff());

	for (int t = steps - 1; t >= 0; t--){
		Dtype* h_t_diff = top_diff + output.offset(t);
		Dtype* c_t_diff = cell_diff + cell.offset(t);
		Dtype* gate_t_diff = gate_diff + gate.offset(t);
		Dtype* pre_gate_t_diff = pre_gate_diff + pre_gate.offset(t);
		//	use h(-1) and c(-1) when t=0
		Dtype* h_t_1_diff = t > 0 ? top_diff + output.offset(t - 1) : h_1.mutable_cpu_diff();
		Dtype* c_t_1_diff = t > 0 ? cell_diff + cell.offset(t - 1) : c_1.mutable_cpu_diff();
		const Dtype* c_t_1_data = t > 0 ? cell_data + cell.offset(t - 1) : c_1.mutable_cpu_data();
		const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
		const Dtype* c_t_data = cell_data + cell.offset(t);
		const Dtype* gate_t_data = gate_data + gate.offset(t);
		for (int n = 0; n < batch_size; n++){
			const bool cont = clip_t ? clip_t[n]>0 : t > 0;
			for (int d = 0; d < hidden_dim; d++){

				//	h_diff(t) += top_diff(t) (branch 1) [from loss]
				//	h_diff(t) += pre_gate_diff(t+1)*U (branch 2) [from next step]
				//	output_gate_diff(t)=h_diff(t)*tanh(c(t))
				const Dtype tanh_c = tanh(c_t_data[d]);
				gate_t_diff[OUTPUT_GATE] = h_t_diff[d] * tanh_c;
				//	c_diff(t) += h_diff(t)*tanh'(c(t))*o(t) (branch 1) [from now step]
				c_t_diff[d] += (h_t_diff[d] * (Dtype(1) - tanh_c*tanh_c)*gate_t_data[OUTPUT_GATE]);
				//	c_diff(t-1) += c_diff(t)*forget_gate_data(t) (branch2) (t>0) [from next step]
				//	note that we pre-compute for t-1 which will be used in for(..t-1)
				c_t_1_diff[d] = cont ? c_t_diff[d] * gate_t_data[FORGET_GATE] : Dtype(0);

				//	forget_gate_diff(t)=c_diff(t)*o(t-1) when t>0
				gate_t_diff[FORGET_GATE] = cont ? c_t_diff[d] * c_t_1_data[d] : Dtype(0);
				//	input_gate_diff(t)=c_diff(t)*c_(t)
				gate_t_diff[INPUT_GATE] = c_t_diff[d] * gate_t_data[CELL_GATE];
				//	cell_gate_diff(t)=c_diff(t)*i(t)
				gate_t_diff[CELL_GATE] = c_t_diff[d] * gate_t_data[INPUT_GATE];

				//	pre_input_gate_diff(t)=input_gate_diff(t)*sigmoid'(pre_input_gate(t))
				//	sigmoid'(x)=sigmoid(x)[1-sigmoid(x)]
				pre_gate_t_diff[INPUT_GATE] = gate_t_diff[INPUT_GATE] *
					gate_t_data[INPUT_GATE] * (Dtype(1) - gate_t_data[INPUT_GATE]);
				//	the same as pre_forget_gate_diff
				pre_gate_t_diff[FORGET_GATE] = gate_t_diff[FORGET_GATE] *
					gate_t_data[FORGET_GATE] * (Dtype(1) - gate_t_data[FORGET_GATE]);
				//	the same as pre_output_gate_diff
				pre_gate_t_diff[OUTPUT_GATE] = gate_t_diff[OUTPUT_GATE] *
					gate_t_data[OUTPUT_GATE] * (Dtype(1) - gate_t_data[OUTPUT_GATE]);
				//	the same as pre_cell_gate_diff
				pre_gate_t_diff[CELL_GATE] = gate_t_diff[CELL_GATE] *
					gate_t_data[CELL_GATE] * (Dtype(1) - gate_t_data[CELL_GATE]);

			}

			if (clipping_threshold > Dtype(0)){
				//	cilp all gates diff
			}

			//	offset for batch_size
			c_t_diff += hidden_dim;
			c_t_data += hidden_dim;
			c_t_1_data += hidden_dim;
			c_t_diff += hidden_dim;
			c_t_1_diff += hidden_dim;
			gate_t_data += 4 * hidden_dim;
			gate_t_diff += 4 * hidden_dim;
			pre_gate_t_diff += 4 * hidden_dim;
		}	// end batch_size

		//	compute h(t-1)_diff in h_to_h
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, hidden_dim, 4 * hidden_dim,
			Dtype(1), pre_gate_diff + pre_gate.offset(t), U, Dtype(0), h_to_h.mutable_cpu_data());

		//	apply h(t-1)_diff to t-1
		//	!!!  h_diff(t) += pre_gate_diff(t+1)*U (branch 2) [from next step]
		for (int n = 0; n < batch_size; n++){
			bool cont = clip_t ? clip_t[n]>0 : t > 0;
			const Dtype* h_to_h_data = h_to_h.cpu_data() + h_to_h.offset(n);
			//	compute h_t_1_diff only when t>0
			if (cont) dragon_add(hidden_dim, h_t_1_diff, h_to_h_data, h_t_1_diff);
		}
	}	//end steps

	// comupte W_diff=pre_gate_diff*bottom_data
	if (param_need_bp[0]){
		dragon_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4 * hidden_dim, input_dim, steps*batch_size,
			Dtype(1), pre_gate_diff, bottom_data, Dtype(1), blobs[0]->mutable_cpu_diff());
	}

	//	compute U_diff=pre_gate_diff(1..T)*top_data
	//	note that Wx(1)+Uh(0) and pre_gate_diff should offset a step
	if (param_need_bp[1]){
		dragon_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4 * hidden_dim, hidden_dim, (steps - 1)*batch_size,
			Dtype(1), pre_gate_diff + pre_gate.offset(1), top_data, Dtype(1), blobs[1]->mutable_cpu_diff());

		//	apply gradient if h(0)!=0 (e.g. use clip)
		dragon_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4 * hidden_dim, hidden_dim, 1,
			Dtype(1), pre_gate_diff, h_1.cpu_data(), Dtype(1), blobs[1]->mutable_cpu_diff());
	}

	//	b_diff=pre_gate_diff
	if (param_need_bp[2]){
		dragon_cpu_gemv<Dtype>(CblasTrans, steps*batch_size, 4 * hidden_dim, Dtype(1),
			pre_gate_diff, bias_multiplier.cpu_data(), Dtype(1), blobs[2]->mutable_cpu_diff());
	}

	//	bottom_diff=pre_gate_diff*W
	if (data_need_bp[0]){
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, steps*batch_size, input_dim, 4 * hidden_dim,
			Dtype(1), pre_gate_diff, W, Dtype(0), bottom[0]->mutable_cpu_diff());
	}

}


INSTANTIATE_CLASS(LSTMLayer);

