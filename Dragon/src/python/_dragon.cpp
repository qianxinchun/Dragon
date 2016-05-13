#pragma warning(disable:4273)
#pragma warning(disable:4244)
#pragma warning(disable:4267)
#pragma warning(disable:4003)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <python.h>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "numpy/arrayobject.h"
#include "common.hpp"
#include "fstream"
#include "iostream"
#include "net.hpp"
#include "utils/io.hpp"
#include "solvers/gradient_solver.hpp"
#include "layer.hpp"
#include "layers/common/python_layer.hpp"
using namespace std;
using namespace boost::python;

typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;
void set_mode_cpu() { Dragon::set_mode(Dragon::CPU); }
void set_mode_gpu() { Dragon::set_mode(Dragon::GPU); }

static void checkFile(const string& filename){
	ifstream file(filename.c_str());
	if (!file.good()){
		file.close();
		throw runtime_error("Could not open file: " + filename);
	}
	file.close();
}

//	load net from net parameter file
boost::shared_ptr<Net<Dtype>> netInit(string param_file, int phase){
	checkFile(param_file);
	boost::shared_ptr<Net<Dtype>> net(new Net<Dtype>(param_file, (Phase)phase));
	return net;
}

boost::shared_ptr<Net<Dtype>> netInitLoad(string param_file, string model_file, int phase){
	checkFile(param_file);
	checkFile(model_file);
	boost::shared_ptr<Net<Dtype>> net(new Net<Dtype>(param_file, (Phase)phase));
	net->copyTrainedLayerFrom(model_file);
	return net;
}

void netSave(const Net<Dtype>& net, string filename){
	NetParameter net_param;
	net.ToProto(&net_param, false);
	writeProtoToBinaryFile(net_param, filename.c_str());
}

//	tuple[blob_ptr,dim1,dim2,....]
object blobReshape(boost::python::tuple args, dict kwargs){
	if (len(kwargs) > 0)
		throw runtime_error("Blob.reshape takes no keyword args");
	Blob<Dtype>* self = extract<Blob<Dtype>*>(args[0]);
	vector<int> shape(len(args) - 1);
	for (int i = 1; i < len(args); i++) 
		shape[i - 1] = extract<int>(args[i]);
	self->reshape(shape);
	return object();
}

typedef vector<boost::shared_ptr<Blob<Dtype>>>  BlobVec;
object addBlob(boost::python::tuple args, dict kwargs){
	if (len(kwargs) > 0)
		throw runtime_error("Blob.add takes no keyword args");
	BlobVec* self = extract<BlobVec*>(args[0]);
	vector<int> shape(len(args) - 1);
	for (int i = 1; i < len(args) - 1; i++)
		shape[i - 1] = extract<int>(args[i]);
	self->push_back(boost::shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
	return object();
}

Solver<Dtype>* getSolverFromFile(const string& filename){
	SolverParameter solver_param;
	readSolverParamsFromTextFileOrDie(filename, &solver_param);
	return new SGDSolver<Dtype>(solver_param);
}


struct NdarrayConverterGenerator{
	template <typename T> struct apply;
};

template<>
struct NdarrayConverterGenerator::apply < Dtype* > {
	struct type{
		PyObject* operator() (Dtype* data) const{
			return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
		}
		const PyTypeObject* get_pytype() { return &PyArray_Type; }
	};
};

struct NdarrayCallPolicies :public default_call_policies{
	typedef NdarrayConverterGenerator result_converter;
	PyObject* postcall(PyObject* pyargs, PyObject* result){
		object pyblob = extract<boost::python::tuple>(pyargs)()[0];
		boost::shared_ptr<Blob<Dtype>> blob = extract < boost::shared_ptr<Blob<Dtype>>>(pyblob);
		void *data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
		Py_DECREF(result);
		const int num_axes = blob->num_axes();
		vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
		PyObject *arr_obj = PyArray_SimpleNewFromData(num_axes, dims.data(), NPY_DTYPE, data);
		Py_INCREF(pyblob.ptr());
		PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj), pyblob.ptr());
		return arr_obj;
	}
};

BOOST_PYTHON_MODULE(_dragon){
	def("set_mode_cpu", &set_mode_cpu);
	def("set_mode_gpu", &set_mode_gpu);
	def("set_device",&Dragon::set_device);
	def("set_random_seed", &Dragon::set_random_seed);

	enum_<Phase>("Phase")
		.value("TRAIN", TRAIN)
		.value("TEST", TEST)
		.export_values();

	//	muti-inherition
	//	Python object 'Net' can both use Net<Dtype> functions
	//	and be parsed as C++ Object
	//	the same as Blob,Layer,Solver
	class_< Net<Dtype>, boost::shared_ptr<Net<Dtype>>, boost::noncopyable>("Net", no_init)
		.def("__init__", make_constructor(&netInit))
		.def("__init__", make_constructor(&netInitLoad))
		.def("_forward", &Net<Dtype>::forwardFromTo)
		.def("_backward", &Net<Dtype>::backwardFromTo)
		.def("reshape", &Net<Dtype>::reshape)
		//	overload member functions
		.def("copyFrom", static_cast<void (Net<Dtype>::*)(const string&)>(&Net<Dtype>::copyTrainedLayerFrom))
		.def("shareWith", &Net<Dtype>::shareTrainedLayerWith)
		.def("save", &netSave)
		.add_property("blob_loss_weights", make_function(
		&Net<Dtype>::getBlobLossWeights, return_internal_reference<>()))
		.add_property("_blobs", make_function(
		&Net<Dtype>::getBlobs, return_internal_reference<>()))
		.add_property("layers", make_function(
		&Net<Dtype>::getLayers, return_internal_reference<>()))
		.add_property("_blob_names", make_function(
		&Net<Dtype>::getBlobNames, return_value_policy<copy_const_reference>()))
		.add_property("_layer_names", make_function(
		&Net<Dtype>::getLayerNames,return_value_policy<copy_const_reference>()))
		.add_property("_inputs", make_function(
		&Net<Dtype>::getInputBlobIdx,return_value_policy<copy_const_reference>()))
		.add_property("_outputs", make_function(
		&Net<Dtype>::getOutputBlobIdx, return_value_policy<copy_const_reference>()));

	class_< Blob<Dtype>, boost::shared_ptr<Blob<Dtype>>, boost::noncopyable>("Blob", no_init)
		.add_property("shape", make_function(
		static_cast<const vector<int>& (Blob<Dtype>::*)() const>(
			&Blob<Dtype>::shape),return_value_policy<copy_const_reference>()))
		.add_property("num", &Blob<Dtype>::num)
		.add_property("channels", &Blob<Dtype>::channels)
		.add_property("height", &Blob<Dtype>::height)
		.add_property("width", &Blob<Dtype>::width)
		.add_property("count", static_cast<int (Blob<Dtype>::*)() const>(&Blob<Dtype>::count))
		//	use python function
		.def("reshape", raw_function(&blobReshape))
		.add_property("data", make_function(&Blob<Dtype>::mutable_cpu_data, NdarrayCallPolicies()))
		.add_property("diff", make_function(&Blob<Dtype>::mutable_cpu_diff, NdarrayCallPolicies()));

	class_< Layer<Dtype>, boost::shared_ptr<PythonLayer<Dtype>>, boost::noncopyable>
		("Layer", init<const LayerParameter&>())
		.add_property("blobs", make_function(&Layer<Dtype>::getBlobs, return_internal_reference<>()))
		.def("setup", &Layer<Dtype>::layerSetup)
		.def("reshape", &Layer<Dtype>::reshape)
		.add_property("phase", make_function(&Layer<Dtype>::getPhase));
	register_ptr_to_python<boost::shared_ptr<Layer<Dtype>>>();
	class_<LayerParameter>("LayerParameter", no_init);

	class_<Solver<Dtype>, boost::shared_ptr<Solver<Dtype>>, boost::noncopyable>("Solver", no_init)
		.add_property("net", &Solver<Dtype>::getTrainNet)
		.add_property("test_nets", make_function(&Solver<Dtype>::getTestNets, return_internal_reference<>()))
		.add_property("iter", &Solver<Dtype>::getIter)
		.def("set_iter", &Solver<Dtype>::setIter)
		.def("solve", &Solver<Dtype>::solve)
		.def("step", &Solver<Dtype>::step)
		.def("restore", &Solver<Dtype>::restore)
		.def("snapshot", &Solver<Dtype>::snapshot);

	//	use base class
	class_<SGDSolver<Dtype>, bases<Solver<Dtype>>, boost::shared_ptr<SGDSolver<Dtype>>, boost::noncopyable>
		("SGDSolver", init<string>());
	class_<AdaDeltaSolver<Dtype>, bases<Solver<Dtype>>, boost::shared_ptr<AdaDeltaSolver<Dtype>>, boost::noncopyable>
		("AdaDeltaSolver", init<string>());

	def("get_solver", make_function(&getSolverFromFile, return_value_policy<manage_new_object>()));


	//	e.g used in Net.blobs[xxx]
	class_<vector<boost::shared_ptr<Blob<Dtype> > > >("BlobVec")
		.def(vector_indexing_suite<vector<boost::shared_ptr<Blob<Dtype> > >, true>());
	//	wrapper for vector<Blob<Dtype>*>
	class_<vector<Blob<Dtype>*>>("RawBlobVec")
		.def(vector_indexing_suite<vector<Blob<Dtype>*>, true>());  //allow index_visiting
	class_<vector<int>>("IntVec")
		.def(vector_indexing_suite<vector<int>, true>());
	class_<vector<boost::shared_ptr<Layer<Dtype> > > >("LayerVec")
		.def(vector_indexing_suite<vector<boost::shared_ptr<Layer<Dtype> > >, true>());
	class_<vector<bool> >("BoolVec")
		.def(vector_indexing_suite<vector<bool> >());
	class_<vector<string> >("StringVec")
		.def(vector_indexing_suite<vector<string> >());
	import_array1();
}
