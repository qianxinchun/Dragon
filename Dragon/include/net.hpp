#ifndef NET_HPP
#define NET_HPP
#include "common.hpp"
#include "blob.hpp"
#include "layer.hpp"
template <typename Dtype>
class Net{
public:
	Net(const NetParameter& param, const Net* root_net = NULL);
	Net(const string& param_file, Phase phase, const Net* root_net = NULL);
	virtual ~Net() {}
	static bool stateMeetRule(const NetState& state, const NetStateRule& rule, const string& name);
	static void filterNet(const NetParameter& param,NetParameter* filtered_param);
	void reshape(){
		for (int i = 0; i < layers.size(); i++)
			layers[i]->reshape(bottom_vecs[i], top_vecs[i]);
	}
	void Init(const NetParameter& in_param);
	Dtype forwardFromTo(int start, int end);
	Dtype forwardFrom(int start);
	Dtype forwardTo(int end);
	const vector<Blob<Dtype>*>& forward(Dtype *loss = NULL);
	ResultGroup forwardWithResult();
	void backwardFromTo(int start, int end);
	void backwardFrom(int start);
	void backwardTo(int end);
	void backward();
	Dtype forwardBackward(){
		Dtype loss;
		forward(&loss);
		backward();
		return loss;
	}
	void clearParamDiffs();
	void shareTrainedLayerWith(const Net* other);
	void copyTrainedLayerFrom(const NetParameter& param);
	void copyTrainedLayerFrom(const string& filename);
	const vector<boost::shared_ptr<Layer<Dtype> > >& getLayers() const {return layers;}
	const vector<boost::shared_ptr<Blob<Dtype> > >& getBlobs() const { return blobs; }
	const vector<string>& getLayerNames() const { return layer_names; }
	const vector<int>& getInputBlobIdx() const { return net_input_blob_indices; }
	const vector<int>& getOutputBlobIdx() const { return net_output_blob_indices; }
	const vector<string>& getBlobNames() const { return blobs_name; }
	const vector<Dtype>& getBlobLossWeights() const{ return blobs_loss_weight; }
	const vector<Blob<Dtype>*> getOutputBlobs() const{ return net_output_blobs; }
	const vector<Blob<Dtype>*> getLearnableParams() const{ return learnable_params; }
	const vector<float> getDecayMults() const{ return params_decay; }
	const vector<float> getLrMults() const{ return params_lr; }
	const string& getNetName() const { return name; }
	void ToProto(NetParameter* param, bool write_diff = false) const;
	void shareWeights();
protected:
	const Net* root_net;
	Phase phase;
	string name;
	int memory_used;
	bool debug_info;
	//	result
	//	store layer
	vector<boost::shared_ptr<Layer<Dtype> > > layers;
	vector<string> layer_names;
	vector<bool> layer_need_backward;
	map<string, int> layers_name_idx;
	//	store for blobs
	vector<boost::shared_ptr<Blob<Dtype> > > blobs;
	map<string, int> blobs_name_idx;
	vector<string> blobs_name;
	vector<bool> blobs_need_backward;
	//	store for top blobs
	vector<vector<Blob<Dtype>*> > top_vecs;
	vector<vector<int> > top_id_vecs;
	//	store for bottom blobs
	vector<vector<Blob<Dtype>*> > bottom_vecs;
	vector<vector<int> > bottom_id_vecs;
	vector<vector<bool> > bottoms_need_backward;
	//	store for param
	vector<Dtype> blobs_loss_weight;
	vector<vector<int> > param_id_vecs;
	vector<string> param_display_names;
	vector<pair<int, int> > param_layer_indices;
	vector<boost::shared_ptr<Blob<Dtype> > > param_blobs;
	vector<Blob<Dtype>*> learnable_params;
	map<string, int> param_names_index;
	vector<int> param_owners;
	vector<int> learnable_param_ids;
	vector<float> params_lr;
	vector<bool> has_params_lr;
	vector<float> params_decay;
	vector<bool> has_params_decay;
	//	blob indices for the input and the output of the net
	vector<int> net_input_blob_indices;
	vector<int> net_output_blob_indices;
	vector<Blob<Dtype>*> net_input_blobs;
	vector<Blob<Dtype>*> net_output_blobs;
	void appendTop(const NetParameter& param, const int layer_id, const int top_id,
		std::set<string>* available_blobs, map<string, int>* blob_name_to_idx);
	int appendBottom(const NetParameter& param, const int layer_id, const int bottom_id,
		std::set<string>* available_blobs, map<string, int>* blob_name_to_idx);
	void appendParam(const NetParameter& param, const int layer_id, const int param_id);
};














#endif
