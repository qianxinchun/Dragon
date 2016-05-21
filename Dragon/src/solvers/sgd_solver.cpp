#include "solvers/gradient_solver.hpp"
#include <cmath>
#include "utils/io.hpp"
#include "utils/math.hpp"

template <typename Dtype>
Dtype SGDSolver<Dtype>::getLearningRate(){
	Dtype rate;
	const string lr_policy = param.lr_policy();
	if (lr_policy == "fixed") rate = param.base_lr();
	else if (lr_policy == "step"){
		current_step = iter / param.step_size();
		rate = param.base_lr()*pow(param.gamma(), current_step);
	}
	else if (lr_policy == "exp")
		rate = param.base_lr()*pow(param.gamma(), iter);
	//	lr_{0}/[(1+¦Ã*iter)]^n
	else if (lr_policy == "inv")
		rate = param.base_lr()*pow(Dtype(1) + param.gamma()*iter,-param.power());
	//	trigger different steps referring current iter
	else if (lr_policy == "multistep"){
		if (current_step < param.step_value_size() && iter >= param.step_value(current_step)){
			current_step++;
			LOG(INFO) << "Multi-Step status: Iteration " << iter << ", step = " << current_step;
		}
		rate = param.base_lr()*pow(param.gamma(), current_step);
	}
	//	lr_{0}*[(1-iter/max)]^n
	else if (lr_policy == "poly") 
		rate = param.base_lr()*pow(Dtype(1) - (Dtype(iter / Dtype(param.max_iter()))), param.power());
	else LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
	return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::preSolve(){
	const vector<Blob<Dtype>*> net_params = net->getLearnableParams();
	history.clear(); update.clear(); temp.clear();
	for (int i = 0; i < net_params.size(); i++){
		const vector<int>& shape = net_params[i]->shape();
		history.push_back(boost::shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
		update.push_back(boost::shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
		temp.push_back(boost::shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
	}
}


template <typename Dtype>
void SGDSolver<Dtype>::clipGradients(){
	const Dtype clip = param.clip_gradients();
	if (clip < 0) return;
	const vector<Blob<Dtype>*> net_params = net->getLearnableParams();
	Dtype sumsq_diff = 0;;
   for (int i = 0; i < net_params.size(); i++) sumsq_diff += net_params[i]->sumsq_diff();
	const Dtype L2_diff = sqrt(sumsq_diff);
	if (L2_diff > clip){
		Dtype factor = clip / L2_diff;
		for (int i = 0; i < net_params.size(); i++) net_params[i]->scale_diff(factor);
	}
}

//	normalize for multi batches in a iter(usually is useless)
template <typename Dtype>
void SGDSolver<Dtype>::normalize(int param_id){
	//	?? 
	if (param.iter_size() == 1) return;
	Blob<Dtype>* net_param = net->getLearnableParams()[param_id];
	const Dtype factor = Dtype(1) / param.iter_size();
	switch (Dragon::get_mode()){
		case Dragon::CPU:
			dragon_scal(net_param->count(), factor, net_param->mutable_cpu_diff());
			break;
		case Dragon::GPU:
#ifndef CPU_ONLY
			dragon_gpu_scal(net_param->count(), factor, net_param->mutable_gpu_diff());
			break;
#endif
		default:LOG(FATAL) << "Unknown mode: " << Dragon::get_mode();
	}
}

template <typename Dtype>
void SGDSolver<Dtype>::regularize(int param_id){
	Blob<Dtype>* net_param = net->getLearnableParams()[param_id];
	const Dtype decay_mult = net->getDecayMults()[param_id];
	Dtype weight_decay = param.weight_decay()*decay_mult;
	if (!weight_decay) return;
	string type = param.regularizer();
	switch (Dragon::get_mode()){
		case Dragon::CPU:
			//	diff += decay*data
			if (type == "L2"){
				dragon_axpy(net_param->count(), weight_decay,
					net_param->cpu_data(), net_param->mutable_cpu_diff());
			}
			//	diff += sign(data)
			else if (type == "L1"){
				dragon_cpu_sign(net_param->count(),
					net_param->cpu_data(), temp[param_id]->mutable_cpu_data());
				dragon_axpy(net_param->count(), weight_decay,
					temp[param_id]->cpu_data(), net_param->mutable_cpu_diff());
			}
			else LOG(FATAL) << "Unknown regularizer: " << type;
			break;
		case Dragon::GPU:
			if (type == "L2"){
				dragon_gpu_axpy(net_param->count(), weight_decay,
					net_param->gpu_data(), net_param->mutable_gpu_diff());
			}
			else if (type == "L1"){
				NOT_IMPLEMENTED;
				/*
				dragon_cpu_sign(net_param->count(),
					net_param->cpu_data(), temp[param_id]->mutable_cpu_data());
				dragon_axpy(net_param->count(), weight_decay,
					temp[param_id]->cpu_data(), net_param->mutable_cpu_diff());*/
			}
			else LOG(FATAL) << "Unknown regularizer: " << type;
			break;
		default:LOG(FATAL) << "Unknown mode: " << Dragon::get_mode();
		}
}

template <typename Dtype>
void SGDSolver<Dtype>::computeUpdateValue(int param_id, Dtype rate){
	Blob<Dtype>* net_param = net->getLearnableParams()[param_id];
	const Dtype lr_mult = net->getLrMults()[param_id];
	Dtype momentum = param.momentum();
	Dtype lr = rate*lr_mult;
	switch (Dragon::get_mode()){
	case Dragon::CPU:
		//	store diff for next
		//	history=momentum*history + lr*diff
		dragon_cpu_axpby<Dtype>(net_param->count(), lr, net_param->cpu_diff(),
			momentum, history[param_id]->mutable_cpu_data());
		dragon_copy<Dtype>(net_param->count(), net_param->mutable_cpu_diff(), 
			history[param_id]->cpu_data());
		break;
	case Dragon::GPU:
#ifndef CPU_ONLY
		dragon_gpu_axpby<Dtype>(net_param->count(), lr, net_param->gpu_diff(),
			momentum, history[param_id]->mutable_gpu_data());
		dragon_gpu_copy<Dtype>(net_param->count(), net_param->mutable_gpu_diff(),
			history[param_id]->gpu_data());
#endif
		break;
	default:LOG(FATAL) << "Unknown mode: " << Dragon::get_mode();
	}
}

template <typename Dtype>
void SGDSolver<Dtype>::applyUpdate(){
	CHECK(Dragon::get_root_solver());
	Dtype rate = getLearningRate();
	if (param.display() && iter%param.display() == 0)
#ifdef USE_PYTHON
		cout << "Iteration " << iter << ", lr = " << rate << endl;
#else 
		LOG(INFO) << "Iteration " << iter << ", lr = " << rate;
#endif
	clipGradients();
	vector<Blob<Dtype>*> net_params = net->getLearnableParams();
	for (int i = 0; i < net_params.size(); i++){
		normalize(i);
		regularize(i);
		computeUpdateValue(i, rate);
		net_params[i]->update();
	}
}

template <typename Dtype>
void SGDSolver<Dtype>::snapshotSolveStateToBinary(const string& filename){
	SolverState state;
	state.set_iter(iter);
	state.set_learned_net(filename);
	state.set_current_step(current_step);
	state.clear_history();

	string state_filename = snapshotFilename(".state");
	LOG(INFO) << "Snapshot state to binary file: " << state_filename;
	writeProtoToBinaryFile(state, state_filename.c_str());
}

template <typename Dtype>
void SGDSolver<Dtype>::snapshotSolverState(const string& filename){
	switch (param.snapshot_format()){
	case SolverParameter_SnapShotFormat_BINARY:
		snapshotSolveStateToBinary(filename);
		break;
	case SolverParameter_SnapShotFormat_HDF5:
		NOT_IMPLEMENTED;
		break;
	default:LOG(FATAL) << "Unknown snapshot mode.";
	}
}
template <typename Dtype>
void SGDSolver<Dtype>::restoreSolverStateFromBinaryProto(const string& filename){
	SolverState state;
	readProtoFromBinaryFile(filename.c_str(), &state);
	iter = state.iter();
	if (state.has_learned_net()){
		NetParameter net_param;
		readProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
		net->copyTrainedLayerFrom(net_param);
	}
	current_step = state.current_step();
	CHECK_EQ(state.history_size(), history.size())
		<< "Incompatible length of history blobs.";
}

INSTANTIATE_CLASS(SGDSolver);