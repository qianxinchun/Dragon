#ifndef IO_HPP
#define IO_HPP
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <google/protobuf/message.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "../common.hpp"
#include "protos/dragon.pb.h"

using google::protobuf::Message;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;

inline bool readProtoFromBinaryFile(const char* filename, Message* proto){
	//	get OS kernel¡®s file descriptor(fd)
	//	successful range:	[0,OPEN_MAX]
	//	replace open(filename, O_RDONLY) as open(filename, O_RDONLY | O_BINARY)
	int fd = _open(filename, O_RDONLY | O_BINARY);
	ZeroCopyInputStream *raw_input = new FileInputStream(fd);
	CodedInputStream *coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(INT_MAX, 536870912);  //  0..512M..2G
	bool success = proto->ParseFromCodedStream(coded_input);
	delete raw_input;
	delete coded_input;
	close(fd);
	return success;
}

inline bool readProtoFromTextFile(const char* filename, Message* proto){
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1) << "File not found:  " << filename;
	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, proto);
	delete input;
	close(fd);
	return success;
}

inline void readProtoFromTextFileOrDie(const char* filename, Message* proto) {
	CHECK(readProtoFromTextFile(filename, proto));
}

inline void readProtoFromBinaryFileOrDie(const char* filename, Message* proto){
	CHECK(readProtoFromBinaryFile(filename, proto));
}

inline void writeProtoToBinaryFile(const Message& proto, const char* filename){
	//	ios::trunc [delete if check file exists]
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	//	use protobuffer interface
	CHECK(proto.SerializeToOstream(&output));
}

inline void readNetParamsFromTextFileOrDie(const string& param_file, NetParameter* param){
	CHECK(readProtoFromTextFile(param_file.c_str(), param))
		<< "Failed to parse NetParameter file.";
}

inline void readSolverParamsFromTextFileOrDie(const string& param_file, SolverParameter* param){
	CHECK(readProtoFromTextFile(param_file.c_str(), param))
		<< "Failed to parse SolverParameter file: " << param_file;
}

#endif

