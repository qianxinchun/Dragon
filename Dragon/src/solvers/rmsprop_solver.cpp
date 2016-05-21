#include "solvers/gradient_solver.hpp"

template <typename Dtype>
void RMSPropSolver<Dtype>::computeUpdateValue(int param_id, Dtype rate){
	Blob<Dtype>* net_param = net->getLearnableParams()[param_id];
	const Dtype lr_mult = net->getLrMults()[param_id];
	Dtype eps = param.delta();
	Dtype momentum = param.momentum();
	Dtype lr = rate*lr_mult;
	const int count = net_param->count();
	switch (Dragon::get_mode()){
	case Dragon::CPU:
		NOT_IMPLEMENTED;
		break;
	case Dragon::GPU:
#ifndef CPU_ONLY
		rmspropUpdate(count, net_param->mutable_gpu_diff(),history[param_id]->mutable_gpu_data(), momentum, eps, lr);
#endif
		break;
	default:LOG(FATAL) << "Unknown mode: " << Dragon::get_mode();
	}
}

template <typename Dtype>
void RMSPropSolver<Dtype>::applyUpdate(){
	CHECK(Dragon::get_root_solver());
	Dtype rate = getLearningRate();
	//	AdaDelta do not need base lr
	if (param.display() && iter%param.display() == 0)
		cout << "Iteration " << iter << ", lr = "<<rate << endl;
	clipGradients();
	vector<Blob<Dtype>*> net_params = net->getLearnableParams();
	for (int i = 0; i < net_params.size(); i++){
		normalize(i);
		regularize(i);
		computeUpdateValue(i, rate);
		net_params[i]->update();
	}
}

INSTANTIATE_CLASS(RMSPropSolver);