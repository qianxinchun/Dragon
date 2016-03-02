#include "include/common.hpp"
#include "solver_include/solver.hpp"
#include "solver_include/gradient_solver.hpp"
#include "include/upgrade.hpp"
#include "include/signal_handler.hpp"
#include "boost/lexical_cast.hpp"
#include <boost/algorithm/string.hpp>   
#include "layer_include/layer_factory.hpp"
#include "include/dragon_thread.hpp"
#pragma warning(disable:4099)
//	define format(name , default value, help string)
DEFINE_string(gpu, "",
	"Optional; run in GPU mode on given device IDs separated by ','."
	"Use '-gpu all' to run on all available GPUs. The effective training "
	"batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
	"The solver definition protocol buffer text file.");
DEFINE_string(model, "",
	"The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
	"Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
	"Optional; the pretrained weights to initialize finetuning, "
	"separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
	"The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
	"Optional; action to take when a SIGINT signal is received: "
	"snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
	"Optional; action to take when a SIGHUP signal is received: "
	"snapshot, stop or none.");
typedef int(*FUNC)();
typedef map<string, FUNC> ArgFactory;
ArgFactory arg_factory;

#define RegisterArgFunction(func) \
class Registerer_##func { \
 public: \
  Registerer_##func() { \
    arg_factory[#func] = &func; \
  } \
};\
Registerer_##func g_registerer_##func

static FUNC getArgFunction(const string& name){
	if (arg_factory.count(name)) return  arg_factory[name];
	else LOG(FATAL) << "Unknown action: " << name;
}

static void get_gpus(vector<int>* gpus){
	//	select all gpus
	if (FLAGS_gpu == "all"){
		int cnt = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&cnt));
#endif
		for (int i = 0; i < cnt; i++) gpus->push_back(i);
	}
	//	select specific gpus(split by ",")
	else if (FLAGS_gpu.size()){
		vector<string> strings;
		boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); i++)
			gpus->push_back(boost::lexical_cast<int>(strings[i]));
	}
	//	select nothing
	else CHECK_EQ(gpus->size(), 0);
}

SolverAction getRequestedAction(const string& flag){
	if (flag == "stop") return SolverAction::STOP;
	if (flag == "snapshot") return SolverAction::SNAPSHOT;
	return SolverAction::NONE;
	LOG(FATAL) << "Invaild signal effec: " << flag;
}


int train(){
	//FLAGS_solver = "G:/Dataset/cifar10/solver.prototxt";
	FLAGS_solver = "G:/Dataset/mood/solver.prototxt";
	FLAGS_snapshot = "G:/Dataset/mood/snapshot/quick_iter_12150.state";
	//FLAGS_snapshot = "G:/Dataset/cifar10/snapshot/quick_iter_3000.state";
	CHECK_GT(FLAGS_solver.size(), 0)<< "Need a solver to be specified.";
	CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
		<< "snapshot and weights can not be specified both.";
	SolverParameter solver_param;
	readSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);

	//	add device id from solver param if necessary
	if (FLAGS_gpu.size() == 0 &&
		solver_param.solver_mode()==SolverParameter_SolverMode_GPU){
		if (solver_param.has_device_id())
			FLAGS_gpu = "" + boost::lexical_cast<string>(solver_param.device_id());
		else FLAGS_gpu = "" + boost::lexical_cast<string>(0);
	}

	//	select gpus use FLAG
	vector<int> gpus;
	get_gpus(&gpus);
	//	use CPU
	if (gpus.size() == 0){
		LOG(INFO) << "Use CPU.";
		//	set root manager and mode
		Dragon::set_mode(Dragon::CPU);
	}
	//	use GPU
	else{
		ostringstream msg;
		for (int i = 0; i < gpus.size(); i++) msg << (i ? "," : "") << gpus[i];
		LOG(INFO) << "Use GPUs: " << msg.str() << " .";
		solver_param.set_device_id(gpus[0]);
		Dragon::set_device(gpus[0]);
		Dragon::set_mode(Dragon::GPU);
		Dragon::set_solver_count(gpus.size());
	}
	
	//	simple but not use Solver Factory
	boost::shared_ptr<Solver<float>> solver(new SGDSolver<float>(solver_param));
	//	default: interrupt-stop/hangup-snapshot
	SignalHandler handler(getRequestedAction(FLAGS_sigint_effect),
		getRequestedAction(FLAGS_sighup_effect));

	//	bind to solver
	solver->setActionFunction(handler.getActionFunction());

	//	resume
	if (FLAGS_snapshot.size()){
		LOG(INFO) << "Resume from: " << FLAGS_snapshot;
		solver->restore(FLAGS_snapshot.c_str());
	}
	//	pre-train
	else if (FLAGS_weights.size()){
		//CopyLayers(...)
	}

	if (gpus.size() > 1){
		NOT_IMPLEMENTED;
	}else{
		LOG(INFO) << "Start Optimization.";
		solver->solve();
	}

	LOG(INFO) << "Optimization Done.";
	return 0;
}

RegisterArgFunction(train);
void globalInit(int* argc, char*** argv){
	gflags::ParseCommandLineFlags(argc, argv, true);
	google::InitGoogleLogging(*(argv)[0]);
	google::LogToStderr();
}
int main(int argc,char* argv[]){
	//	Initialize Google's logging library.
	globalInit(&argc, &argv);
	//	if (argc == 2) return getArgFunction(string(argv[1]))();
	train();
	while (1) {}
}
