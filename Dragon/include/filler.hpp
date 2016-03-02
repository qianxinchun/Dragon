#ifndef FILLER_HPP
#define FILLER_HPP
#include "include/dragon.pb.h"
#include "include/common.hpp"
#pragma warning(disable:4715)
template<typename Dtype>
class Filler{
public:
	Filler(const FillerParameter& param) :param(param) {}
	virtual void fill(Blob<Dtype>* ptr_blob) = 0;
protected:
	FillerParameter param;
};

template<typename Dtype>
class ConstantFiller : public Filler < Dtype > {
public:
	ConstantFiller(const FillerParameter& param) :Filler<Dtype>(param) {}
	virtual void fill(Blob<Dtype> *ptr_blob){
		Dtype *base_data = ptr_blob->mutable_cpu_data();
		const int count = ptr_blob->count();
		CHECK(count);
		const Dtype val = param.value();
		for (int i = 0; i < count; i++) base_data[i] = val;
	}
};

template<typename Dtype>
class UniformFiller :public Filler < Dtype > {
public:
	UniformFiller(const FillerParameter& param) :Filler<Dtype>(param) {}
	virtual void fill(Blob<Dtype>* ptr_blob){
		CHECK(ptr_blob->count());
		dragon_rng_uniform(ptr_blob->count(), param.min(), param.max(), ptr_blob->mutable_cpu_data());
	}

};

template<typename Dtype>
class GaussianFiller :public Filler < Dtype > {
public:
	GaussianFiller(const FillerParameter& param) :Filler<Dtype>(param) {}
	virtual void fill(Blob<Dtype>* ptr_blob){
		CHECK(ptr_blob->count());
		dragon_rng_gaussian<Dtype>(ptr_blob->count(), param.mean(), param.std(), ptr_blob->mutable_cpu_data());
		// Sparse Gaussian Initialization has not been implemented yet
	}

};

template<typename Dtype>
Filler<Dtype>* getFiller(const FillerParameter& param){
	const string& type = param.type();
	if (type == "constant") return new ConstantFiller<Dtype>(param);
	if (type == "gaussian") return new GaussianFiller<Dtype>(param);
}
#endif