#ifndef PYTHON_LAYER_HPP
#define PYTHON_LAYER_HPP

#include <boost/python.hpp>
#include "../../layer.hpp"
using namespace boost::python;

template <typename Dtype>
class PythonLayer :public Layer < Dtype > {
public:
	PythonLayer(PyObject* self, const LayerParameter& param) :
		Layer<Dtype>(param), self(handle<>(borrowed(self))) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		if (phase == TRAIN&&Dragon::get_solver_count() > 1 && !shareInParallel()){
			LOG(FATAL) << "PythonLayer is not implemented in multi-GPU training.";
		}
		self.attr("param_str") = boost::python::str(param.python_param().param_str());
		self.attr("setup")(bottom, top);
	}
	virtual void reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		self.attr("reshape")(bottom, top);
	}
	virtual bool shareInParallel() { return param.python_param().share_in_parallel(); }
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		self.attr("forward")(bottom, top);
	}
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
		const vector<Blob<Dtype>*> &bottom){
		self.attr("backward")(top, data_need_bp, bottom);
	}
private:
	object self;
};

#endif