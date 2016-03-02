#ifndef LAYER_FACTORY_HPP
#define LAYER_FACTORY_HPP
#include "map"
#include "include/common.hpp"
#include "include/dragon.pb.h"

//	declare but not define in case mutual reference
//	because this hpp will be included in layer.hpp

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerFactory{
public:
	//	NEW_FUNC type can point to a (new XXXLayer<Dtype>(param))
	typedef boost::shared_ptr< Layer<Dtype> > (*NEW_FUNC)(const LayerParameter& );
	typedef map<string, NEW_FUNC> Factory;
	//	trick: static member's declaration and definition and initialzation and get_wrapper
	//		   cab be write as a simple way
	static Factory& getFactory() {
		static Factory* factory = new Factory();
		return *factory;
	}
	//	use map<string,pointer> to store the xxxLayer's new function pointer
	static void reg(const string& type, NEW_FUNC new_pointer){
		Factory& factory= getFactory();
		CHECK_EQ(factory.count(type), 0)
			<< "This layer type had been registered.";
		factory[type] = new_pointer;
	}

	//	search for xxxLayer's new function
	//	and use it to "new" a instance object
	static boost::shared_ptr< Layer<Dtype> > createLayer(const LayerParameter& param){
		CHECK_EQ(getFactory().count(param.type()), 1)
			<< "Unknown layer type: " << param.type();
		if (Dragon::get_root_solver())
			LOG(INFO) << "Create " << param.type() << "Layer: " << param.name();
		//	it can be considered as " new xxxLayer<Dtype>(param) "
		return getFactory()[param.type()](param);
	}
private:
	//	trick: forbid LayerRegistry be instantiated
	//		   it should always work in static mode
	LayerFactory() {}
};

//	an external portal for layer registing used in #define(....)
template <typename Dtype>
class LayerRegister{
public:
	typedef boost::shared_ptr< Layer<Dtype> >(*NEW_FUNC)(const LayerParameter&);
	LayerRegister(const string& type, NEW_FUNC new_pointer){
		LayerFactory<Dtype>::reg(type, new_pointer);
	}
};

//	stuff different numerical pointer into Factory
#define REGISTER_LAYER_CREATOR(type,creator)	\
	static LayerRegister<float> g_creator_f_##type(#type,creator<float>);	\
	static LayerRegister<double> g_creator_d_##type(#type,creator<double>)	\

//	construct a function with parameter list
//	we can not let the pointer point to (new ....) directly
//	because (new ....) with parameter list will trigger instantiation
#define REGISTER_LAYER_CLASS(type)	\
	template <typename Dtype>	\
	boost::shared_ptr< Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param){	\
		return boost::shared_ptr< Layer<Dtype> >(new type##Layer<Dtype>(param));	\
	}	\
	REGISTER_LAYER_CREATOR(type,Creator_##type##Layer)
#endif


