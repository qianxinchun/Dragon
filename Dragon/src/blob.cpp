#include "include/blob.hpp"
#include "include/synced_mem.hpp"
template <typename Dtype>
void Blob<Dtype>::reshape(int num,int channels,int height,int width) {
	vector<int> shape(4);
	shape[0] = num;shape[1] = channels;
	shape[2] = height;shape[3] = width;
	reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::reshape(const BlobShape& blob_shape) {
	vector<int> shape(blob_shape.dim_size());
	for (int i = 0; i < shape.size(); i++) shape[i] = blob_shape.dim(i);
	reshape(shape);
}

template<typename Dtype>
void Blob<Dtype>::reshape(vector<int> shape){
	count_ = 1;
	shape_.resize(shape.size());
	for (int i = 0; i < shape.size(); ++i) {
		count_ *= shape[i];
		shape_[i] = shape[i];
	}
	//	if new count ≠ old capacity
	//	recycle and allocate memory again
	if (count_ > capacity_) {
		capacity_ = count_;
		data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	}
}

template<typename Dtype>
void Blob<Dtype>::reshapeLike(const Blob& blob){
	reshape(blob.shape_);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const{
	CHECK(data_);
	return (const Dtype*)data_->cpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype *data){
	CHECK(data_);
	data_->set_cpu_data(data);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const{
	CHECK(data_);
	return (const Dtype*)data_->gpu_data();
}

template<typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype *data){
	CHECK(data_);
	data_->set_gpu_data(data);
}

template<typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const{
	CHECK(diff_);
	return (const Dtype*)diff_->cpu_data();
}

template<typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff()const {
	CHECK(diff_);
	return (const Dtype*)diff_->gpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data(){
	CHECK(data_);
	return (Dtype*)(data_->mutable_cpu_data());
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data(){
	CHECK(data_);
	return (Dtype*)data_->mutable_gpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff(){
	CHECK(diff_);
	return (Dtype*)diff_->mutable_cpu_data();
}

template<typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff(){
	CHECK(diff_);
	return (Dtype*)diff_->mutable_gpu_data();
}
template<> void Blob<unsigned int>::update() { NOT_IMPLEMENTED; }
template<> void Blob<int>::update() { NOT_IMPLEMENTED; }


template<typename Dtype>
void Blob<Dtype>::update(){
	switch(data_->head()){
	case SyncedMemory::SyncedHead::HEAD_AT_CPU:
		dragon_axpy(count_, Dtype(-1), cpu_diff(), mutable_cpu_data());
		break;
	case SyncedMemory::SyncedHead::HEAD_AT_GPU:
	case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
		dragon_gpu_axpy<Dtype>(count_, Dtype(-1), gpu_diff(), mutable_gpu_data());
#endif
		break;
	default:
		// UNINITIALIZED JUST DO NOTHING
		;
	}
}

template<> unsigned int Blob<unsigned int>::asum_data() { NOT_IMPLEMENTED; return 0; }
template <> int Blob<int>::asum_data() { NOT_IMPLEMENTED; return 0; }

template <> unsigned int Blob<unsigned int>::sumsq_diff() const { NOT_IMPLEMENTED; return 0; }
template <> int Blob<int>::sumsq_diff() const { NOT_IMPLEMENTED;return 0;}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
	Dtype sumsq;
	const Dtype* diff;
	if (!diff_) { return 0; }
	switch (diff_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		diff = cpu_diff();
		sumsq = dragon_cpu_dot(count_, diff, diff);
		break;
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		diff = gpu_diff();
		sumsq = dragon_gpu_dot(count_, diff, diff);
		break;
#endif
	case SyncedMemory::UNINITIALIZED:
		return 0;
	default:
		LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}
	return sumsq;
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {NOT_IMPLEMENTED;}
template <> void Blob<int>::scale_diff(int scale_factor) {NOT_IMPLEMENTED;}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
	Dtype* diff;
	if (!diff_) { return; }
	switch (diff_->head()) {
	case SyncedMemory::HEAD_AT_CPU:
		diff = mutable_cpu_diff();
		dragon_scal(count_, scale_factor, diff);
		return;
	case SyncedMemory::HEAD_AT_GPU:
	case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
		diff = mutable_gpu_diff();
		dragon_gpu_scal(count_, scale_factor, diff);
		return;
#endif
	case SyncedMemory::UNINITIALIZED:return;
	default:LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
	}
}


template<typename Dtype>
Dtype Blob<Dtype>::asum_data(){
	switch(data_->head()){
	case SyncedMemory::SyncedHead::HEAD_AT_CPU:
		return dragon_cpu_asum(count_,mutable_cpu_data());
	case SyncedMemory::SyncedHead::HEAD_AT_GPU:
	case SyncedMemory::SyncedHead::SYNCED:
#ifndef CPU_ONLY
		return dragon_gpu_asum(count_, (Dtype*)data_->gpu_data());
#endif
	case SyncedMemory::SyncedHead::UNINITIALIZED:
		return 0;
	default:LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
	}
}

template<typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool need_reshape = true){
	//copy shape
	if (need_reshape){
		vector<int> shape;
		shape.resize(proto.shape().dim_size());
		for (int i = 0; i < shape.size(); i++)
			shape[i] = proto.shape().dim(i);
		reshape(shape);
	}
	if (proto.data_size()>0){
		CHECK_EQ(proto.data_size(), count());
		Dtype *data = mutable_cpu_data();
		for (int i = 0; i < count_; i++) data[i] = proto.data(i);
	}
	if (proto.diff_size()>0){
		CHECK_EQ(proto.diff_size(), count());
		Dtype *diff = mutable_cpu_diff();
		for (int i = 0; i < count_; i++) diff[i] = proto.diff(i);
	}
}

template<typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff){
	proto->clear_shape();
	proto->clear_data();
	proto->clear_diff();
	//do not use proto->shape() cause it is a const method
	for (int i = 0; i < shape_.size(); i++)  proto->mutable_shape()->add_dim(shape_[i]);
	const Dtype *data = cpu_data();
	const Dtype *diff = cpu_diff();
	for (int i = 0; i < count_; i++)  proto->add_data(data[i]);
	if (write_diff)
		for (int i = 0; i < count_; i++)  proto->add_diff(diff[i]);
}


//	explicit instantiation for class
//	more info see http://bbs.csdn.net/topics/380250382

INSTANTIATE_CLASS(Blob);
template class Blob < int > ;
template class Blob < unsigned int > ;



