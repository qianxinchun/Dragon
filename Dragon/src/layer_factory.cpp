#include "layer_factory.hpp"
#include "layers/vision_layers.hpp"
#include "layers/data_layers.hpp"
#include "layers/loss_layers.hpp"
#include "layers/neuron_layers.hpp"
#include "layers/common_layers.hpp"
//#include "application_layers.hpp"

REGISTER_LAYER_CLASS(Data);
//REGISTER_LAYER_CLASS(AppData);
//REGISTER_LAYER_CLASS(ImageData);
//REGISTER_LAYER_CLASS(Prediction);
REGISTER_LAYER_CLASS(Convolution);
REGISTER_LAYER_CLASS(Pooling);
REGISTER_LAYER_CLASS(Reshape);
REGISTER_LAYER_CLASS(ROIPooling);
REGISTER_LAYER_CLASS(InnerProduct);
REGISTER_LAYER_CLASS(Accuracy);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);
REGISTER_LAYER_CLASS(Softmax);
REGISTER_LAYER_CLASS(SmoothL1Loss);
REGISTER_LAYER_CLASS(L2Loss);
REGISTER_LAYER_CLASS(ReLU);
REGISTER_LAYER_CLASS(Split);
REGISTER_LAYER_CLASS(BatchNorm);
REGISTER_LAYER_CLASS(Dropout);
REGISTER_LAYER_CLASS(Concat);
REGISTER_LAYER_CLASS(LRN);
REGISTER_LAYER_CLASS(Crop);



template <typename Dtype>
boost::shared_ptr<Layer<Dtype>> GetPythonLayer(const LayerParameter& param){
	// C++ execute PyCode, init PyInterpreter
	Py_Initialize();
	try{
		object moudle = import(param.python_param().module().c_str());
		object layer = moudle.attr(param.python_param().layer().c_str())(param);
		// use shared_ptr to point to a PyObject
		return extract<boost::shared_ptr<PythonLayer<Dtype>>>(layer)();
	}catch (error_already_set){
		PyErr_Print();
		throw;
	}
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
