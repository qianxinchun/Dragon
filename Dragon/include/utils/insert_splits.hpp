#ifndef INSERT_SPLITS_HPP
#define INSERT_SPLITS_HPP
#include <string>
#include "protos/dragon.pb.h"
using namespace std;

void insertSplits(const NetParameter& param, NetParameter* splitted_param);

string splitBlobName(const string& layer_name, const string& blob_name,
	const int blob_idx, const int split_idx);

string splitLayerName(const string& layer_name, const string& blob_name, const int blob_idx);

void configureSplitLayer(const string& layer_name, const string& blob_name,
	const int blob_idx, const int split_count, const float loss_weight, LayerParameter* param);

#endif