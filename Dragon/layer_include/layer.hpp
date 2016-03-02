#ifndef LAYER_HPP
#define LAYER_HPP
#include "include/dragon.pb.h"
#include "include/common.hpp"
#include "include/blob.hpp"
//	disable warning due to INSTANTIATE for forward/backward gpu
//	but not implement in cpp file(actually in cu file)
#pragma warning(disable:4661)
template <typename Dtype>
class Layer
{
public:
	// copy the phase and previous train parameter 
	Layer(const LayerParameter& param) :param(param), is_shared(false){
		phase = param.phase();
		if (param.blobs_size() > 0){
			blobs.resize(param.blobs_size());
			for (int i = 0; i < blobs.size(); i++){
				blobs[i].reset(new Blob<Dtype>());
				blobs[i]->FromProto(param.blobs(i));
			}
		}
	}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {}
	void initMutex() { forward_mutex.reset(new boost::mutex()); }
	void lock() {if (is_shared) forward_mutex->lock();}
	void unlock() { if (is_shared) forward_mutex->unlock();}
	vector < boost::shared_ptr<Blob<Dtype>>> &getBlobs() { return this->blobs; }
	virtual bool allowForceBackward(const int bottom_idx) const { return true; }
	void setShared(bool is_shared){
		CHECK(shareInParallel() || !is_shared) << "Layer does not support sharing.";
		this->is_shared = is_shared;
	}
	virtual bool shareInParallel() { return false; }
	void setParamNeedBp(const int param_id, const bool is_need){
		if (param_need_bp.size() <= param_id) param_need_bp.resize(param_id + 1, true);
		param_need_bp[param_id] = is_need;
	}
	Dtype getLoss(const int top_idx) const {return (loss.size() > top_idx) ? loss[top_idx] : Dtype(0);}
	void setLoss(const int top_idx, const Dtype val){
			if (loss.size() <= top_idx) loss.resize(top_idx + 1);
			loss[top_idx] = val;
	}
	void setLossWeight(const vector<Blob<Dtype>*>& top){
		const int num = param.loss_weight_size();
		if (num){
			CHECK_EQ(top.size(), num)
				<< "The number of loss_weights must equal to the number of top blobs.";
			for (int top_id = 0; top_id < top.size(); top_id++){
				const Dtype loss_weight = param.loss_weight(top_id);
				if (loss_weight == Dtype(0)) continue;
				setLoss(top_id, loss_weight);
				const int cnt = top[top_id]->count();
				//	store the diff's value as loss_weight (0-invaild/1-vaild)
				//	this only work in LossLayer
				//	because LossLayer need not diff to backward propogation
				Dtype *loss_multiplier = top[top_id]->mutable_cpu_diff();
				dragon_set(cnt, loss_weight, loss_multiplier);
			}
		}

	}
	// do most initial work before forward and backward
	void setup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		initMutex();
		layerSetup(bottom, top);
		reshape(bottom, top);
		setLossWeight(top);
	}
	Dtype forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		lock();
		Dtype tot_loss = 0;
		reshape(bottom, top);
		switch (Dragon::get_mode()){
			case Dragon::CPU:
				forward_cpu(bottom, top);
				//	check if need loss
				for (int top_id = 0; top_id < top.size(); top_id++){
					if (!getLoss(top_id)) continue;
					const int cnt = top[top_id]->count();
					//	data represent loss
					//	loss_weights(0/1) represent whether use loss
					const Dtype* data = top[top_id]->cpu_data();
					const Dtype* loss_weights = top[top_id]->cpu_diff();
					tot_loss += dragon_cpu_dot<Dtype>(cnt, data, loss_weights);
				}
				break;
			case Dragon::GPU:
				forward_gpu(bottom, top);
#ifndef CPU_ONLY
				for (int top_id = 0; top_id < top.size(); ++top_id) {
					if (!getLoss(top_id)) continue;
					const int cnt = top[top_id]->count();
					const Dtype* data = top[top_id]->gpu_data();
					const Dtype* loss_weights = top[top_id]->gpu_diff();
					tot_loss+=dragon_gpu_dot(cnt, data, loss_weights);
				}
#endif
				break;
			default:
				LOG(FATAL) << "Unknown running device mode.";
			}
		unlock();
		return tot_loss;
	}
	void backward(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
		switch (Dragon::get_mode()){
			case Dragon::CPU:
				backward_cpu(top, data_need_bp, bottom);
				break;
			case Dragon::GPU:
				backward_gpu(top, data_need_bp, bottom);
				break;
			default:
				LOG(FATAL) << "Unknown running device mode.";
		}
	}
	virtual void ToProto(LayerParameter* param, bool write_diff = false);
	virtual ~Layer() {}
protected:
	LayerParameter param;
	Phase phase;
	//	a layer can be shared by other net
	//	caffe use forward_mutex to avoid conflictions
	bool is_shared;
	vector < boost::shared_ptr< Blob<Dtype> > > blobs;
	vector < bool > param_need_bp;
	boost::shared_ptr<boost::mutex> forward_mutex;
	vector<Dtype> loss;
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) = 0;
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		forward_cpu(bottom, top);
	}
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
		backward_gpu(top, data_need_bp, bottom);
	}
};

template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff){
	param->Clear();
	//	copy configurations
	param->CopyFrom(this->param);
	param->clear_blobs();
	//	copy learnable param blobs
	for (int i = 0; i < blobs.size(); i++) blobs[i]->ToProto(param->add_blobs(), write_diff);
}

#endif

