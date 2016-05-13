# ifndef BASE_CONV_LAYER_HPP
# define BASE_CONV_LAYER_HPP

# include "../../layer.hpp"
# include "../../utils/im2col.hpp"

template <typename Dtype>
class BaseConvolutionLayer :public Layer<Dtype>{
public:
	BaseConvolutionLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	Blob<int> kernel_shape;
	Blob<int> stride;
	Blob<int> pad;
	Blob<int> conv_input_shape;
	vector<int> col_buffer_shape;
	vector<int> output_shape;
	vector<int> bottom_shape;
	Blob<Dtype> col_buffer;
	Blob<Dtype> bias_multiplier;
	int num_axes, num_spatial_axes;
	int bottom_dim, top_dim;
	int channels_axis;
	int num, channels, group, out_spatial_dim;
	int weight_offset, num_output;
	bool bias_term, is_1x1, force_nd_im2col;

	int num_kernels_im2col, num_kernels_col2im;
	int conv_in_channels, conv_out_channels;
	int conv_out_spatial_dim, kernel_dim;
	int col_offset, output_offset;
	virtual bool reverseDimensions() = 0;
	virtual void computeOutputShape() = 0;
	void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col = false);
	void forward_cpu_bias(Dtype* output, const Dtype* bias);
	void backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);
	void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype *weights);
	void backward_cpu_bias(Dtype* bias, const Dtype* input);
	//	can not use STUB_GPU
#ifndef CPU_ONLY
	void forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output, bool skip_im2col = false);
	void forward_gpu_bias(Dtype* output, const Dtype* bias);
	void backward_gpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input);
	void backward_gpu_bias(Dtype* bias, const Dtype* input);
	void weight_gpu_gemm(const Dtype* input, const Dtype* output, Dtype *weights);
#endif

private:
	//	wrap im2col using param in this class
	//	data is 3D(channels,height,width), col_buff is 3D(channels
	void conv_im2col_cpu(const Dtype* data, Dtype* col_buff){
		// only implements conv2D
		if (!force_nd_im2col&&num_spatial_axes == 2){
			//	im2col transform the input into the form which is convenient for convolution
			//	use conv_xxx cause dimensions could reverse in reshape(), we need dynamic input
			im2col_cpu(data, conv_in_channels, conv_input_shape.cpu_data()[1], conv_input_shape.cpu_data()[2],
				kernel_shape.cpu_data()[0], kernel_shape.cpu_data()[1], pad.cpu_data()[0], pad.cpu_data()[1],
				stride.cpu_data()[0], stride.cpu_data()[1], col_buff);
		}
	}
	void conv_col2im_cpu(const Dtype* col_buff, Dtype* data){
		if (!force_nd_im2col&&num_spatial_axes == 2){
			col2im_cpu(col_buff, conv_in_channels, conv_input_shape.cpu_data()[1], conv_input_shape.cpu_data()[2],
				kernel_shape.cpu_data()[0], kernel_shape.cpu_data()[1], pad.cpu_data()[0], pad.cpu_data()[1],
				stride.cpu_data()[0], stride.cpu_data()[1], data);
		}
	}
#ifndef CPU_ONLY
	void conv_im2col_gpu(const Dtype* data, Dtype* col_buff){
		if (!force_nd_im2col&&num_spatial_axes == 2){
			im2col_gpu(data, conv_in_channels, conv_input_shape.cpu_data()[1], conv_input_shape.cpu_data()[2],
				kernel_shape.cpu_data()[0], kernel_shape.cpu_data()[1], pad.cpu_data()[0], pad.cpu_data()[1],
				stride.cpu_data()[0], stride.cpu_data()[1], col_buff);
		}
	}
	void conv_col2im_gpu(const Dtype* col_buff, Dtype* data){
		if (!force_nd_im2col&&num_spatial_axes == 2){
			col2im_gpu(col_buff, conv_in_channels, conv_input_shape.cpu_data()[1], conv_input_shape.cpu_data()[2],
				kernel_shape.cpu_data()[0], kernel_shape.cpu_data()[1], pad.cpu_data()[0], pad.cpu_data()[1],
				stride.cpu_data()[0], stride.cpu_data()[1], data);
		}
	}
#endif
};

# endif