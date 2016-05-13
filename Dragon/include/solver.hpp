#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "net.hpp"

template <typename Dtype>
class Solver{
public:
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
	//	implemented by different ways
	virtual void applyUpdate() = 0;
	boost::shared_ptr<Net<Dtype>> getTrainNet() { return net; }
	const vector<boost::shared_ptr<Net<Dtype> > >& getTestNets() { return test_nets; }
	int getIter() { return iter; }
	void setIter(int iter) { this->iter = iter; }
protected:
	const Solver* root_solver;
	SolverParameter param;
	boost::shared_ptr<Net<Dtype>> net;
	vector<boost::shared_ptr<Net<Dtype> > > test_nets;
	int iter,current_step;
};
#endif