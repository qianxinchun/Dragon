#ifndef SIGNAL_HANDLER_HPP
#define SIGNAL_HANDLER_HPP

#include "include/dragon.pb.h"
#include "solver_include/solver.hpp"
class SignalHandler {
public:
	SignalHandler(SolverAction SIGINT_action,SolverAction SIGHUP_action);
	~SignalHandler();
	ActionCallBack getActionFunction();
private:
	SolverAction checkForSignals() const;
	SolverAction SIGINT_action;
	SolverAction SIGHUP_action;
};


#endif
