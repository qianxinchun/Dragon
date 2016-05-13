#include <fstream>
#include "utils/io.hpp"
#include "solver.hpp"

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver = NULL)
	:param(param), root_solver(root_solver){
	init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const  Solver* root_solver = NULL)
	:root_solver(root_solver){
	readSolverParamsFromTextFileOrDie(param_file, &this->param);
	init(param);
}

template <typename Dtype>
string Solver<Dtype>::snapshotFilename(const string extension){
	string filename(param.snapshot_prefix());
	char buffer[20];
	_snprintf(buffer, 20, "_iter_%d", iter);
	return filename + buffer + extension;
}

template <typename Dtype>
void Solver<Dtype>::checkSnapshotWritePermission(){
	if (Dragon::get_root_solver() && param.snapshot_interval()){
		CHECK(param.has_snapshot_prefix()) <<
			"Must specify snapshot_prefix if need snapshot";
		string filename = snapshotFilename(".tmp");
		ofstream ofs(filename.c_str());
		//	test if write sucessfully
		if (ofs.good()){
			ofs.close();
			//	delete tmp file
			remove(filename.c_str());
		}else LOG(FATAL) << "Can not write to snapshot_prefix: "<< param.snapshot_prefix() << ".";
	}
}

template <typename Dtype>
void Solver<Dtype>::initTrainNet(){
	const int num_train_nets = param.has_net_file() + param.has_net_param() + 
		param.has_train_net_file() + param.has_train_net_param();
	const string& field_names = "net_file/net_param/train_net_file/train_net_param";
	CHECK_GE(num_train_nets, 1)
		<< "Must specify a train net using one of these fields: " << field_names;
	CHECK_LE(num_train_nets, 1)
		<< "Must not use more than one of these fields to specify a train net: " << field_names;
	NetParameter net_param;
	if (param.has_train_net_param()){
		LOG_IF(INFO, Dragon::get_root_solver())
			<< "Create train net from train net param.";
		net_param.CopyFrom(param.train_net_param());
	}
	if (param.has_net_param()){
		LOG_IF(INFO, Dragon::get_root_solver())
			<< "Create train net from net param.";
		net_param.CopyFrom(param.net_param());
	}
	if (param.has_train_net_file()){
		LOG_IF(INFO, Dragon::get_root_solver())
			<< "Create train net from train net file: " << param.train_net_file();
		readNetParamsFromTextFileOrDie(param.train_net_file(), &net_param);
	}
	if (param.has_net_file()){
		LOG_IF(INFO, Dragon::get_root_solver())
			<< "Create train net from net file: " << param.net_file();
		readNetParamsFromTextFileOrDie(param.net_file(), &net_param);
	}
	NetState net_state;
	net_state.set_phase(TRAIN);
	//	update state
	if(net_param.has_state())
		net_state.MergeFrom(net_param.state());		//	low prio (default: not set)
	if (param.has_train_state())
		net_state.MergeFrom(param.train_state());	//	high prio (default: not set)
	//	merge to net_state
	net_param.mutable_state()->CopyFrom(net_state);
	//	create and init net after parsing net_param
	if (Dragon::get_root_solver())
		net.reset(new Net<Dtype>(net_param));
	else net.reset(new Net<Dtype>(net_param, root_solver->net.get()));
}

template <typename Dtype>
void Solver<Dtype>::initTestNets(){
	CHECK(Dragon::get_root_solver())
		<< "Test nets can only run in root solver.";
	const int num_general_nets = param.has_net_param() + param.has_net_file();
	CHECK_LE(num_general_nets, 1)
		<< "Net file and net param can not be specified at same time.";
	const int num_test_params = param.test_net_param_size();
	const int num_test_files = param.test_net_file_size();
	const int num_test_nets = num_test_params + num_test_files;
	if (num_general_nets)
		CHECK_GE(param.test_iter_size(), num_test_nets) << "Test iter must be specified for each test net.";
	else CHECK_EQ(param.test_iter_size(), num_test_nets)<< "Test iter must be specified for each test net.";
	const int num_general_instances = param.iter_size() - num_test_nets;
	const int num_test_instances = param.test_iter_size();
	if (param.test_state_size()){
		CHECK_EQ(param.test_state_size(), num_test_instances)
			<< "Test state must be un-specified or specified once per test net.";
	}
	if (num_test_instances)
		CHECK_GT(param.test_interval(), 0) << "Test interval must be specified(not zero).";
	int net_id = 0;
	vector<string> sources(num_test_instances);
	vector<NetParameter> net_params(num_test_instances);
	//	handle test nets
	for (int i = 0; i < num_test_params; i++){
		sources[net_id] = "test_net_param";
		net_params[net_id++].CopyFrom(param.test_net_param(i));
	}
	for (int i = 0; i < num_test_files; i++){
		sources[net_id] = "test_net_file: " + param.test_net_file(i);
		readNetParamsFromTextFileOrDie(param.test_net_file(i), &net_params[net_id++]);
	}
	//	handle general net
	const int remain_test_nets = num_test_instances - net_id;
	if (param.has_net_param()){
		sources[net_id] = "net_param";
		net_params[net_id++].CopyFrom(param.net_param());
	}
	if (param.has_net_file()){
		sources[net_id] = "net_file: " + param.net_file();
		readNetParamsFromTextFileOrDie(param.net_file(), &net_params[net_id++]);
	}
	test_nets.resize(num_test_instances);
	for (int i = 0; i < num_test_instances; i++){
		NetState net_state;
		net_state.set_phase(TEST);
		if (net_params[i].has_state())
			net_state.MergeFrom(net_params[i].state());	//	low prio (default: not set)
		if (param.test_state_size())
			net_state.MergeFrom(param.test_state(i));	//	high prio (default: not set)
		net_params[i].mutable_state()->CopyFrom(net_state);
		LOG(INFO) << "Create test net #" << i << ": from " << sources[i];
		if (Dragon::get_solver_count()) test_nets[i].reset(new Net<Dtype>(net_params[i]));
		else test_nets[i].reset(root_solver->test_nets[i].get());
		//test_nets[i]->setDebugInfo(param.debug_info());
	}

}

template <typename Dtype>
void Solver<Dtype>::init(const SolverParameter& param){
	CHECK(Dragon::get_root_solver() || root_solver)
		<< "Root solver need be set for all non-root solvers.";
	LOG_IF(INFO, Dragon::get_root_solver()) << "Initialize solver from parameters: "
		<< endl << param.DebugString();
	CHECK_GE(param.average_loss(), 1) << "Average loss should greater equal than 1.";
	checkSnapshotWritePermission(); 
	//	set seed for random_generator if necessary
	if (Dragon::get_root_solver() && param.random_seed() >= 0)
		Dragon::set_random_seed(param.random_seed());
	//	create and init a train net
	initTrainNet();
	if (Dragon::get_root_solver()){
		initTestNets();
		LOG(INFO) << "Solver initialization done.";
	}
	iter = current_step = 0;
}

template <typename Dtype>
void Solver<Dtype>::solve(const char* resume_file){
	CHECK(Dragon::get_root_solver());
	LOG(INFO) << "Solve: " << net->getNetName();
	LOG(INFO) << "Learning rate policy: " << param.lr_policy();
	if (resume_file){
		LOG(INFO) << "Restore previous solver status from" << resume_file;
		restore(resume_file);
	}
	step(param.max_iter() - iter);
	if (param.snapshot_after_train() &&
		(!param.snapshot_interval() || iter%param.snapshot_interval() != 0))
		snapshot();

	if (param.display() && iter%param.display() == 0){
		Dtype loss;
		net->forward(&loss);
		LOG(INFO) << "Iteration " << iter << ", loss = " << loss;
	}

	if (param.test_interval() && iter%param.test_interval() == 0)
		testAll();

	LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
string Solver<Dtype>::snapshotToBinary(){
	string filename = snapshotFilename(".model");
	LOG(INFO) << "Snapshot model to binary file: " << filename;
	NetParameter net_param;
	net->ToProto(&net_param, param.snapshot_diff());
	writeProtoToBinaryFile(net_param, filename.c_str());
	return filename;
}

template <typename Dtype>
void Solver<Dtype>::snapshot(){
	//	only root solver can snapshot ?
	CHECK(Dragon::get_root_solver());
	string filename;
	switch (param.snapshot_format()){
		case SolverParameter_SnapShotFormat_BINARY:
			filename = snapshotToBinary();
			break;
		case  SolverParameter_SnapShotFormat_HDF5:
			NOT_IMPLEMENTED;
			break;
		default:LOG(FATAL) << "Unknown snapshot mode.";
	}
	//	should not implement here
	snapshotSolverState(filename);
}

template <typename Dtype>
void Solver<Dtype>::test(int net_id){
	CHECK(Dragon::get_root_solver());
	LOG(INFO) << "Train iteration: " << iter << ", Test net #" << net_id << ": ";
	//	share params 
	test_nets[net_id]->shareTrainedLayerWith(net.get());
	vector<Dtype> test_score;
	vector<int> output_id;
	Net<Dtype>* test_net = test_nets[net_id].get();
	Dtype loss = 0;
	//	scan for all batches
	for (int i = 0; i < param.test_iter(net_id);i++){
		Dtype iter_loss;
		const vector<Blob<Dtype>*>& result = test_net->forward(&iter_loss);
		if (param.test_compute_loss()) loss += iter_loss;
		if (i == 0){
			for (int j = 0; j < result.size(); j++){
				const Dtype* base_output = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); k++){
					//	fill output position
					test_score.push_back(base_output[k]);
					output_id.push_back(j);
				}
			}
		}else{
			int idx = 0;
			for (int j = 0; j < result.size(); j++){
				const Dtype* base_output = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); k++) test_score[idx++] += base_output[k];
			}
		}
	}
	if (param.test_compute_loss()){
		loss /= param.test_iter(net_id);
		LOG(INFO) << "Test loss: " << loss;
	}
	for (int i = 0; i < test_score.size(); i++){
		const int blob_idx = test_net->getOutputBlobIdx()[output_id[i]];
		const string& output_name = test_net->getBlobNames()[blob_idx];
		const Dtype loss_weight = test_net->getBlobLossWeights()[blob_idx];
		ostringstream msg;
		//	per batch
		const Dtype mean_score = test_score[i] / param.test_iter(net_id);
		if (loss_weight)
			msg << " (* " << loss_weight << " = " << loss_weight*mean_score << " loss)";
		LOG(INFO) << "		Test net output #" << i << "(" << output_name << "): "
			<< mean_score << msg.str();
	}
} 

template <typename Dtype>
void Solver<Dtype>::testAll(){
	for (int net_id = 0; net_id < test_nets.size(); net_id++) test(net_id);
}

template <typename Dtype>
void Solver<Dtype>::step(int iters){
	const int start_iter = iter, stop_iter = iter + iters;
	int average_loss = param.average_loss();
	vector<Dtype> loss_vec;
	Dtype smoothed_loss = 0;
	while (iter < stop_iter){
		// clear accumulative diffs in last iter
		net->clearParamDiffs();
		//	cross vaildation or test
		if (param.test_interval() && iter%param.test_interval() == 0 &&Dragon::get_root_solver()){
			// check if need test before train
			if ((iter == 0 && param.test_before_train()) || iter != 0) testAll();
		}
		const bool display = param.display() && iter%param.display() == 0;
		Dtype loss = 0;
		for (int i = 0; i < param.iter_size(); i++) loss += net->forwardBackward();
		loss /= param.iter_size();
		//	smoothed_loss use the last num_(average_loss) iters to average
		//	default use last iter(average_loss=1)
		if (loss_vec.size() < average_loss){
			//	fill
			loss_vec.push_back(loss);
			int size = loss_vec.size();
			smoothed_loss = (smoothed_loss*(size - 1) + loss) / size;
		}
		else{
			//replace
			int idx = (iter - start_iter) % average_loss;
			//cout << (loss - loss_vec[idx]) << endl;
			smoothed_loss += ((loss - loss_vec[idx]) / average_loss);
			loss_vec[idx] = loss;
		}
		if (display){
#ifdef USE_PYTHON
			cout << "Iteration " << iter << ", loss = " << smoothed_loss << endl;
#else
			LOG_IF(INFO, Dragon::get_root_solver()) << "Iteration " << iter << ", loss = " << smoothed_loss;
#endif
			int score_idx = 0;
			const vector<Blob<Dtype>*>& result = net->getOutputBlobs();
			for (int i = 0; i < result.size(); i++){
				const Dtype* res_vec = result[i]->cpu_data();
				const string& output_name = net->getBlobNames()[net->getOutputBlobIdx()[i]];
				const Dtype loss_weight = net->getBlobLossWeights()[net->getOutputBlobIdx()[i]];
				for (int j = 0; j < result[i]->count(); j++){
					ostringstream msg;
					if (loss_weight)
						msg << " (* " << loss_weight << " = " << loss_weight*res_vec[j] << " loss)";
#ifdef USE_PYTHON
					    cout << "		Train net output #" << i << "(" << output_name << "): " << res_vec[j] << msg.str() << endl;
#else 
						LOG(INFO) << "		Train net output #" << i << "(" << output_name << "): "<< res_vec[j] << msg.str();
#endif
				}
			}
		}
		applyUpdate();
		iter++;
		// snapshot if at the time or necessary
		if ((param.snapshot_interval() && iter%param.snapshot_interval() == 0 && Dragon::get_root_solver()))
			snapshot();
	}
}

template <typename Dtype>
void Solver<Dtype>::restore(const char* filename){
	CHECK(Dragon::get_root_solver());
	string state_filename(filename);
	//	HDF5
	// NOT_IMPLEMENTED;
	//	PROTO
	restoreSolverStateFromBinaryProto(state_filename);
}


INSTANTIATE_CLASS(Solver);