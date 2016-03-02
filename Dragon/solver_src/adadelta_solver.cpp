#include "solver_include/gradient_solver.hpp"
template <typename Dtype>
void AdaDeltaSolver<Dtype>::computeUpdateValue(int param_id, Dtype rate){
	Blob<Dtype>* net_param = net->getLearnableParams()[param_id];
	const Dtype lr_mult = net->getLrMults()[param_id];
	Dtype eps = param.delta();
	Dtype momntum = param.momentum();
	// adadelta will ignore base_lr
	Dtype lr = lr_mult;
	const int count = net_param->count();
	switch (Dragon::get_mode()){
	case Dragon::CPU:
		//	history store for E[g^2]
		//	update store for E[delta^2]
		//	history=momentum*history + (1-momentum)*(diff^2)
		//	1. compute diff^2 in temp
		dragon_powx<Dtype>(count, net_param->cpu_diff(), Dtype(2), temp[param_id]->mutable_cpu_data());
		//	2. compute history
		dragon_cpu_axpby<Dtype>(count, Dtype(1) - momntum, temp[param_id]->cpu_data(),
				momntum, history[param_id]->mutable_cpu_data());
		//	3. compute RMS[history] as denominator in temp
		dragon_set<Dtype>(count, eps, temp[param_id]->mutable_cpu_data());
		dragon_axpy<Dtype>(count, Dtype(1), history[param_id]->cpu_data(),temp[param_id]->mutable_cpu_data());
		dragon_powx<Dtype>(count, temp[param_id]->cpu_data(), Dtype(0.5), temp[param_id]->mutable_cpu_data());
		//	4. compute diff/RMS[history] in diff
		dragon_div<Dtype>(count, net_param->cpu_diff(), temp[param_id]->cpu_data(), net_param->mutable_cpu_diff());
		//	5. compute RMS[update] as numerator in temp
		dragon_set<Dtype>(count, eps, temp[param_id]->mutable_cpu_data());
		dragon_axpy<Dtype>(count, Dtype(1), update[param_id]->cpu_data(), temp[param_id]->mutable_cpu_data());
		dragon_powx<Dtype>(count, temp[param_id]->cpu_data(), Dtype(0.5), temp[param_id]->mutable_cpu_data());
		//	6. compute diff*RMS[update] in diff
		dragon_mul<Dtype>(count, net_param->cpu_diff(), temp[param_id]->cpu_data(), net_param->mutable_cpu_diff());
		//	7. compute final diff^2 in temp
		dragon_powx<Dtype>(count, net_param->cpu_diff(), Dtype(2), temp[param_id]->mutable_cpu_data());
		//	8. compute update
		dragon_cpu_axpby<Dtype>(count, (1 - momntum), temp[param_id]->cpu_data(),
			momntum, update[param_id]->mutable_cpu_data());
		//	9. apply learning rate
	    dragon_scal<Dtype>(count, lr, net_param->mutable_cpu_diff());
		break;
	case Dragon::GPU:
#ifndef CPU_ONLY
		dragon_gpu_powx<Dtype>(count, net_param->gpu_diff(), Dtype(2), temp[param_id]->mutable_gpu_data());
		//	2. compute history
		dragon_gpu_axpby<Dtype>(count, Dtype(1) - momntum, temp[param_id]->gpu_data(),
			momntum, history[param_id]->mutable_gpu_data());
		//	3. compute RMS[history] as denominator in temp
		dragon_gpu_set<Dtype>(count, eps, temp[param_id]->mutable_gpu_data());
		dragon_gpu_axpy<Dtype>(count, Dtype(1), history[param_id]->gpu_data(), temp[param_id]->mutable_gpu_data());
		dragon_gpu_powx<Dtype>(count, temp[param_id]->gpu_data(), Dtype(0.5), temp[param_id]->mutable_gpu_data());
		//	4. compute diff/RMS[history] in diff
		dragon_gpu_div<Dtype>(count, net_param->gpu_diff(), temp[param_id]->gpu_data(), net_param->mutable_gpu_diff());
		//	5. compute RMS[update] as numerator in temp
		dragon_gpu_set<Dtype>(count, eps, temp[param_id]->mutable_gpu_data());
		dragon_gpu_axpy<Dtype>(count, Dtype(1), update[param_id]->gpu_data(), temp[param_id]->mutable_gpu_data());
		dragon_gpu_powx<Dtype>(count, temp[param_id]->gpu_data(), Dtype(0.5), temp[param_id]->mutable_gpu_data());
		//	6. compute diff*RMS[update] in diff
		dragon_gpu_mul<Dtype>(count, net_param->gpu_diff(), temp[param_id]->gpu_data(), net_param->mutable_gpu_diff());
		//	7. compute final diff^2 in temp
		dragon_gpu_powx<Dtype>(count, net_param->gpu_diff(), Dtype(2), temp[param_id]->mutable_gpu_data());
		//	8. compute update
		dragon_gpu_axpby<Dtype>(count, Dtype(1) - momntum, temp[param_id]->gpu_data(),
			momntum, update[param_id]->mutable_gpu_data());
		//	9. apply learning rate
		dragon_gpu_scal<Dtype>(count, lr, net_param->mutable_gpu_diff());
#endif
		break;
	default:LOG(FATAL) << "Unknown mode: " << Dragon::get_mode();
	}
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::applyUpdate(){
	CHECK(Dragon::get_root_solver());
	Dtype rate = getLearningRate();
	//	AdaDelta do not need base lr
	if (param.display() && iter%param.display() == 0)
		LOG(INFO) << "Iteration " << iter << ", lr = AdaDelta";
	clipGradients();
	vector<Blob<Dtype>*> net_params = net->getLearnableParams();
	for (int i = 0; i < net_params.size(); i++){
		normalize(i);
		regularize(i);
		computeUpdateValue(i, rate);
		net_params[i]->update();
	}
}

INSTANTIATE_CLASS(AdaDeltaSolver);