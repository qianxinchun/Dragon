#include "include/net.hpp"
#include "include/io.hpp"
#include "include/upgrade.hpp"
template <typename Dtype>
class AppSolver{
public:
	void initTestNet(string model_file, string net_file){
		//	restore learned_net to learned_net_param
		readProtoFromBinaryFile(model_file.c_str(), &learned_net_param);
		//	read text_net to net_param
		readNetParamsFromTextFileOrDie(net_file, &net_param);
		NetState net_state;
		net_state.set_phase(TEST);
		if (net_param.has_state())
			net_state.MergeFrom(net_param.state());	//	low prio (default: not set)
		net_param.mutable_state()->CopyFrom(net_state);
		LOG(INFO) << "Create test net " << ":  " << net_file;
		LOG(INFO) << "		 Source net :  " << model_file;
		net.reset(new Net<Dtype>(net_param));
		//	copy parameters
		net->copyTrainedLayerFrom(learned_net_param);
	}
	void test(){
		LOG(INFO) << "Start Test";
		vector<Blob<Dtype>*> bottom_vec;
		const vector<Blob<Dtype>*>& result = test_net->forward(bottom_vec, &iter_loss);
	}
private:
	NetParameter learned_net_param, net_param;
	boost::shared_ptr<Net<Dtype>> net;
};
void globalInit(int* argc, char*** argv){
	gflags::ParseCommandLineFlags(argc, argv, true);
	google::InitGoogleLogging(*(argv)[0]);
	google::LogToStderr();
}
int main(int argc, char* argv[]){
	globalInit(&argc, &argv);
	AppSolver < float > solver;
	string model_file = "G:/Dataset/cifar10/snapshot/quick_iter_10000.model";
	string net_file = "G:/Dataset/cifar10/net_train_test.prototxt";
	solver.initTestNet(model_file, net_file);
	while (1) {}
}