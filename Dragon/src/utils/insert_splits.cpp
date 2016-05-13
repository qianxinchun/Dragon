#include <map>
#include "utils/insert_splits.hpp"
#include "common.hpp"

string splitBlobName(const string& layer_name, const string& blob_name,
	const int blob_idx, const int split_idx){
	ostringstream split_blob_name;
	// fill into a string
	split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
		<< "_split_" << split_idx;
	return split_blob_name.str();
}

string splitLayerName(const string& layer_name, const string& blob_name, const int blob_idx){
	ostringstream split_layer_name;
	split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx << "_split";
	return split_layer_name.str();
}

void configureSplitLayer(const string& layer_name, const string& blob_name,
	const int blob_idx, const int split_count, const float loss_weight, LayerParameter* param){
	param->Clear();
	param->add_bottom(blob_name);;
	param->set_name(splitLayerName(layer_name, blob_name, blob_idx));
	param->set_type("Split");
	for (int i = 0; i < split_count; i++){
		param->add_top(splitBlobName(layer_name, blob_name, blob_idx, i));
		//	LossLayer
		if (loss_weight){
			if (i == 0) param->add_loss_weight(loss_weight);
			else param->add_loss_weight(0);
		}
	}
}

void insertSplits(const NetParameter& param, NetParameter* splitted_param){
	splitted_param->CopyFrom(param);
	splitted_param->clear_layer();
	//	pair<layer_idx,blob_idx>
	map<string, pair<int, int>> blob_name_to_last_top_idx;
	map<pair<int, int>, pair<int, int>> bottom_idx_to_source_top_idx;
	map<pair<int, int>, int> top_idx_to_bottom_count;
	map<pair<int, int>, float> top_idx_to_loss_weight;
	map<pair<int, int>, int> top_idx_to_bottom_split_idx;
	map<int, string> layer_idx_to_layer_name;
	layer_idx_to_layer_name[-1] = "input";
	//	scan and stuff all input blobs into a virtual layer named as "input" at -1
	//  input blobs do not belong to any layers and we stuff them into a virtual layer
	//	usually use for viewing a Net(e.g: examples\cifar10\cifar10_full.prototxt
	//	input: "data"      ***  ¡û_¡û  specify it as a temporary data blob ***
	//	input_shape{	   ***  ¡û_¡û  specify it as shape***
	//		dim: 1
	//		dim : 3
	//		dim : 32
	//		dim : 32
	//	}
	//	pay attention: input blobs should not use in train/test prototxt
	//	because they are not specified vaild data sources
	//	you can regard them as viewing toys
	for (int i = 0; i < param.input_size(); i++){
		const string& blob_name = param.input(i);
		blob_name_to_last_top_idx[blob_name] = make_pair(-1, i);
	}
	for (int i = 0; i < param.layer_size(); i++){
		const LayerParameter& layer_param = param.layer(i);
		//	bind layer idx to layer name
		layer_idx_to_layer_name[i] = layer_param.name();
		//	a layer has several bottom blobs(e.g DataLayer)
		for (int j = 0; j < layer_param.bottom_size(); j++){
			const string& blob_name = layer_param.bottom(j);
			//	ensure that all bottom blobs must have the same name as one top blob
			if (!blob_name_to_last_top_idx.count(blob_name)){
				LOG(FATAL) << "Unknown bottom blob: " << blob_name
					<< " at layer: " << layer_param.name() << ".";
			}
			const pair<int, int>& bottom_idx = make_pair(i, j);
			const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
			//	a bottom's name must be as same as one top's name
			//	find a bottom's parent top (<- backward direction)
			//	note that top name must declare before bottom name
			//	or a bottom will bind to layer_{-1}
			bottom_idx_to_source_top_idx[bottom_idx] = top_idx;
			top_idx_to_bottom_count[top_idx]++;
		}
		// update top name's position for following bottom names
		for (int j = 0; j < layer_param.top_size(); j++){
			const string& blob_name = layer_param.top(j);
			blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
		}
		const int last_loss = min(layer_param.loss_weight_size(), layer_param.top_size());
		//	only work in LossLayer
		for (int j = 0; j < last_loss; j++){
			const string& blob_name = layer_param.top(j);
			//	updated before
			const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
			top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
			//	from loss(top) backward to bottom 
			if (top_idx_to_loss_weight[top_idx]) top_idx_to_bottom_count[top_idx]++;
		}
	}
	//	special case: data blob shared by other blobs in the virtual layer
	//	split it also
	for (int i = 0; i < param.input_size(); i++){
		const int split_count = top_idx_to_bottom_count[make_pair(-1, i)];
		if (split_count > 1){
			//	"input"
			const string& layer_name = layer_idx_to_layer_name[-1];
			const string& blob_name = param.input(i);
			//	push_back a new param
			LayerParameter* split_layer_param = splitted_param->add_layer();
			const float kZeroLossWeight = 0;
			configureSplitLayer(layer_name, blob_name, i, split_count, kZeroLossWeight, split_layer_param);
		}
	}
	for (int i = 0; i < param.layer_size(); i++){
		//	push_back a new param
		LayerParameter* layer_param = splitted_param->add_layer();
		layer_param->CopyFrom(param.layer(i));
		for (int j = 0; j < layer_param->bottom_size(); j++){
			//  call the top before bottom
			const pair<int, int>& top_idx = bottom_idx_to_source_top_idx[make_pair(i, j)];
			//	check top's count
			const int split_count = top_idx_to_bottom_count[top_idx];
			if (split_count > 1){
				// previous layer_name
				const string& layer_name = layer_idx_to_layer_name[top_idx.first];
				const string& blob_name = layer_param->bottom(j);
				// e.g: conv1 => conv1_conv1_0_split_0
				//	once used then ++ for next
				layer_param->set_bottom(j, splitBlobName(layer_name, blob_name, top_idx.second, 
					top_idx_to_bottom_split_idx[top_idx]++));
			}
		}
		for (int j = 0; j < layer_param->top_size(); j++){
			const pair<int, int>& top_idx = make_pair(i, j);
			const int split_count = top_idx_to_bottom_count[top_idx];
			if (split_count > 1){
				//	now layer_name
				const string& layer_name = layer_idx_to_layer_name[top_idx.first];
				const string& blob_name = layer_param->top(j);
				//	add a split layer
				LayerParameter *split_layer_param = splitted_param->add_layer();
				const float loss_weight = top_idx_to_loss_weight[top_idx];
				configureSplitLayer(layer_name, blob_name, j, split_count, loss_weight,split_layer_param);
				if (loss_weight){
					layer_param->clear_loss_weight();
					// loss as bottom should split from 1 ???
					top_idx_to_bottom_split_idx[top_idx]++;
				}
			}
		}
	}
}