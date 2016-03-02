#include "include/signal_handler.hpp"
#include "signal.h"
#include "csignal"
#pragma warning(disable:4800)

//	interrupt signal
static volatile sig_atomic_t got_sigint = false;
//	hang-up signal
static volatile sig_atomic_t got_sighup = false;
static bool already_hooked_up = false;
void handle_signal(int signal){
	switch (signal)
	{
		case SIGBREAK:
			got_sighup = true;
			break;
		case SIGINT:
			got_sigint = true;
			break;
	}
}

void hookHandler(){
	if (already_hooked_up)
		LOG(FATAL) << "Try to hookup signal handlers more than once.";
	already_hooked_up = true;
	//	install handler function
	if (signal(SIGBREAK, handle_signal) == SIG_ERR) 
		LOG(FATAL) << "Cannot install SIGBREAK handler.";
	if (signal(SIGINT, handle_signal) == SIG_ERR)
		LOG(FATAL) << "Cannot install SIGINT handler.";
}

void unhookHandler() {
	if (already_hooked_up) {
		if (signal(SIGBREAK, SIG_DFL) == SIG_ERR)
			LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
		if (signal(SIGINT, SIG_DFL) == SIG_ERR)
			LOG(FATAL) << "Cannot uninstall SIGINT handler.";
		already_hooked_up = false;
	}
}

//	get signal status and inverse it
bool gotSIGINT() {
	bool result = got_sigint;
	got_sigint = false;
	return result;
}

bool gotSIGHUP() {
	bool result = got_sighup;
	got_sighup = false;
	return result;
}

SignalHandler::SignalHandler(SolverAction SIGINT_action,SolverAction SIGHUP_action) :
	SIGINT_action(SIGINT_action),SIGHUP_action(SIGHUP_action) {
	hookHandler();
}

SignalHandler::~SignalHandler() {unhookHandler();}

SolverAction SignalHandler::checkForSignals() const {
	if (gotSIGHUP()) return SIGHUP_action;
	if (gotSIGINT()) return SIGINT_action;
	return SolverAction::NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallBack SignalHandler::getActionFunction() {
	return boost::bind(&SignalHandler::checkForSignals, this);
}

