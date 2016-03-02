#ifndef UPGRADE_HPP
#define UPGRADE_HPP
#include "include/io.hpp"
#include "include/net.hpp"
// upgrade proto param from old version
inline void upgradeNetAsNeed(const string& param_file, NetParameter* param){
	NOT_IMPLEMENTED;
}
inline void readNetParamsFromTextFileOrDie(const string& param_file,NetParameter* param){
	CHECK(readProtoFromTextFile(param_file.c_str(), param))
		<< "Failed to parse NetParameter file.";
	upgradeNetAsNeed(param_file, param);
}

inline void readSolverParamsFromTextFileOrDie(const string& param_file,SolverParameter* param){
	CHECK(readProtoFromTextFile(param_file.c_str(), param))
		<< "Failed to parse SolverParameter file: " << param_file;

}
#endif