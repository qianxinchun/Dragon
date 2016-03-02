#ifndef GRADIENT_SOLVER_HPP
#define GRADIENT_SOLVER_HPP
#include "solver_include/solver.hpp"
template <typename Dtype>
class SGDSolver :public Solver < Dtype > {
public:
	SGDSolver(const SolverParameter& param) :Solver<Dtype>(param)	{ preSolve(); }
	SGDSolver(const string& param_file) :Solver<Dtype>(param_file)	{ preSolve(); }
protected:
	vector<boost::shared_ptr<Blob<Dtype>>> history, update, temp;
	void preSolve();
	Dtype getLearningRate();
	virtual void clipGradients();
	virtual void applyUpdate();
	virtual void normalize(int param_id);
	virtual void regularize(int param_id);
	virtual void computeUpdateValue(int param_id, Dtype rate);
	virtual void snapshotSolverState(const string& filename);
	virtual void snapshotSolveStateToBinary(const string& filename);
	virtual void restoreSolverStateFromBinaryProto(const string& filename);
};

template <typename Dtype>
class AdaDeltaSolver :public SGDSolver < Dtype > {
public:
	AdaDeltaSolver(const SolverParameter& param) :SGDSolver<Dtype>(param)	{ }
	AdaDeltaSolver(const string& param_file) :SGDSolver<Dtype>(param_file)	{ }
protected:
	virtual void computeUpdateValue(int param_id, Dtype rate);
	virtual void applyUpdate();
};







#endif