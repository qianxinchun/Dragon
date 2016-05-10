#include "data_include/data_transformer.hpp"
#include "include/io.hpp"
#include "include/common.hpp"
template <typename Dtype>
DataTransformer<Dtype>::DataTransformer(const  TransformationParameter& param, Phase phase):
	param(param), phase(phase)
{
	//	normally, we get mean_value from mean_file
	if (param.has_mean_file()){
		CHECK_EQ(param.mean_value_size(), 0) << "System wants to use mean_file but specified mean_value.";
		const string& mean_file = param.mean_file();
		LOG(INFO) << "Loading mean file from: " << mean_file;
		BlobProto proto;
		readProtoFromBinaryFileOrDie(mean_file.c_str(), &proto);
		mean_blob.FromProto(proto);
	}
	//	using each channel's mean value
	//	mean_value_size() is between 1 and 3
	if (param.mean_value_size()>0){
		CHECK(param.has_mean_file() == false) << "System wants to use mean_value but specified mean_file.";
		for (int i = 0; i < param.mean_value_size(); i++)
			mean_vals.push_back(param.mean_value(i));
	}
	initRand();
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::inferBlobShape(const Datum& datum){
	const int crop_size = param.crop_size();
	const int channels = datum.channels();
	const int height = datum.height();
	const int width = datum.width();
	CHECK_GT(channels, 0);
	CHECK_GE(height, crop_size);
	CHECK_GE(width,crop_size);
	vector<int> shape(4);
	shape[0] = 1; shape[1] = channels;
	shape[2] = crop_size ? crop_size : height;
	shape[3] = crop_size ? crop_size : width;
	return shape;
}
template<typename Dtype>
void DataTransformer<Dtype>::initRand(){
	const bool must_rand = param.mirror() || (phase == TRAIN && param.crop_size());
	if (must_rand){
		//thread-independent and fixed random-seed
		const unsigned int rng_seed = Dragon::get_random_value();
		ptr_rng.reset(new Dragon::RNG(rng_seed));
	}
}

template<typename Dtype>
int DataTransformer<Dtype>::rand(int n){
	CHECK(ptr_rng);
	CHECK_GT(n, 0);
	rng_t* rng = ptr_rng->get_rng();
	return (*rng)() % n;
}

//	copy the datum to the
template<typename Dtype>
void DataTransformer<Dtype>::transform(const Datum& datum, Dtype* shadow_data){
	//	pixel can be compressed as a string
	//	cause each pixel ranges from 0~255 (a char)
	const string& data = datum.data();
	const int datum_channels = datum.channels();
	const int datum_height = datum.height();
	const int datum_width = datum.width();
	const int crop_size = param.crop_size();
	const Dtype scale = param.scale();
	const bool need_mirror = param.mirror() && rand(2); // random mirrow
	const bool has_mean_file = param.has_mean_file();
	const bool has_uint8 = data.size() > 0;		//	pixels are compressed as a string
	const bool has_mean_value = mean_vals.size() > 0;
	CHECK_GT(datum_channels, 0);
	CHECK_GE(datum_height, crop_size);
	CHECK_GE(datum_width, crop_size);
	Dtype *mean = NULL;
	if (has_mean_file){
		CHECK_EQ(datum_channels, mean_blob.channels());
		CHECK_EQ(datum_height, mean_blob.height());
		CHECK_EQ(datum_width, mean_blob.width());
		mean = mean_blob.mutable_cpu_data();
	}
	if (has_mean_value){
		CHECK(mean_vals.size() == 1 || mean_vals.size() == datum_channels)
			<< "Channel's mean value must be provided as a single value or as many as channels.";
		//replicate
		if (datum_channels > 1 && mean_vals.size() == 1)
			for (int i = 0; i < datum_channels - 1; i++)
				mean_vals.push_back(mean_vals[0]);
	}
	int h_off = 0, w_off = 0, height = datum_height, width = datum_width;
	if (crop_size){
		height = crop_size;
		width = crop_size;
		//	train phase using random croping
		if (phase == TRAIN){
			h_off = rand(datum_height - height + 1);
			w_off = rand(datum_width - width + 1);
		}
		//	test phase using expected croping
		else{
			h_off = (datum_height - height) / 2;
			w_off = (datum_width - width) / 2;
		}
	}
	Dtype element;
	int top_idx, data_idx;
	//copy datum values to shadow_data-> batch
	for (int c = 0; c < datum_channels; c++){
		for (int h = 0; h < height; h++){
			for (int w = 0; w < width; w++){
				data_idx = (c*datum_height + h_off + h)*datum_width + w_off + w;
				if (need_mirror)	top_idx = (c*height + h)*width + (width - 1 - w); //top_left=top_right
				else	top_idx = (c*height + h)*width + w;
				if (has_uint8){
					//	char type can not cast to Dtype directly
					//	or will generator mass negative number(facing Cifar10)
					element=static_cast<Dtype>(static_cast<uint8_t>(data[data_idx]));
				}
				else element = datum.float_data(data_idx);	//Dtype <- float
				if (has_mean_file) shadow_data[top_idx] = (element - mean[data_idx])*scale;
				else if (has_mean_value) shadow_data[top_idx] = (element - mean_vals[c])*scale;
				else shadow_data[top_idx] = element*scale;
			}
		}
	}
}

template<typename Dtype>
void DataTransformer<Dtype>::transform(const Datum& datum, Blob<Dtype>* shadow_blob){
	const int num = shadow_blob->num();
	const int channels = shadow_blob->channels();
	const int height = shadow_blob->height();
	const int width = shadow_blob->width();
	CHECK_EQ(channels, datum.channels());
	CHECK_GE(num, 1);
	CHECK_LE(height, datum.height()); //allowing crop
	CHECK_LE(width, datum.width());
	Dtype *base_data = shadow_blob->mutable_cpu_data();
	transform(datum, base_data);
}

INSTANTIATE_CLASS(DataTransformer);
