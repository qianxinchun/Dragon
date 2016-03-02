#ifndef SOLVER_HPP
#define SOLVER_HPP
#include "boost/function.hpp"
#include "include/net.hpp"
enum SolverAction{NONE,STOP,SNAPSHOT};
typedef boost::function<SolverAction()> ActionCallBack;

template <typename Dtype>
class Solver{
public:
	class CallBack{
	protected:
		virtual void on_start() = 0;
		virtual void on_gradients_ready() = 0;
		template <typename T>
		friend class Solver;
	};
	Solver(const SolverParameter& param, const Solver* root_solver = NULL);
	Solver(const string& param_file, const  Solver* root_solver = NULL);
	void init(const SolverParameter& param);
	~Solver() { }
	void initTrainNet();
	void initTestNets();
	string snapshotFilename(const string extension);
	void checkSnapshotWritePermission();
	string snapshotToBinary();
	virtual void restoreSolverStateFromBinaryProto(const string& state_file) = 0;
	virtual void snapshotSolverState(const string& filename) = 0;
	void solve(const char* resume_file = NULL);
	void snapshot();
	void restore(const char* filename);
	void test(int net_id);
	void testAll();
	void step(int iters);
	void setActionFunction(ActionCallBack function);
	//	implemented by different ways
	virtual void applyUpdate() = 0;
	SolverAction getRequestedAction();
protected:
	const Solver* root_solver;
	SolverParameter param;
	boost::shared_ptr<Net<Dtype>> net;
	vector<boost::shared_ptr<Net<Dtype> > > test_nets;
	vector<CallBack*> callbacks;
	bool need_early_stopping;
	int iter,current_step;
	ActionCallBack action_request_function;
};
#endif