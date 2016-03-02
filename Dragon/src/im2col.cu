#include "include/im2col.hpp"
#include "alternative/device_alternative.hpp"
#include "include/common.hpp"
template<typename Dtype>
__global__ void im2col_gpu_kernel(const int n,const Dtype* im,const int height, const int width,
	const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
	const int stride_h, const int stride_w, const int col_h, const int col_w,Dtype* col){
	CUDA_KERNEL_LOOP(idx, n){
		const int h_idx = idx / col_w;
		const int im_c = h_idx / col_h;
		const int h = h_idx%col_h;
		const int w = idx%col_w;
		const int c = im_c*kernel_h*kernel_w;
		const int im_h_off = h*stride_h - pad_h;
		const int im_w_off = w*stride_w - pad_w;
		//	compute the first col pos of a roll convolution
		Dtype* col_ptr = col;
		col_ptr += ((c*col_h + h)*col_w + w);
		//	compute the first im pos of a roll convolution
		const Dtype* im_ptr = im;
		im_ptr += ((im_c*height + im_h_off)*width + im_w_off);
		for (int i = 0; i < kernel_h; i++){
			for (int j = 0; j < kernel_w; j++){
				//	compute the current im pos
				int im_h = i + im_h_off;
				int im_w = j + im_w_off;
				*col_ptr = (im_h >= 0 && im_w >= 0 && im_h < height&&im_w < width) ?
					im_ptr[i*width + j] : 0;
				col_ptr += (col_h*col_w);
			}
		}
	}
}

template<typename Dtype>
void im2col_gpu(const Dtype* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* col){
	const int col_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	const int col_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	const int n = (channels*col_h*col_w);
	im2col_gpu_kernel<Dtype> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(
		n, im, height, width, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, col_h,col_w, col);
	CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void col2im_gpu_kernel(const int n,const Dtype* col, const int height, const int width,
	const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
	 const int stride_h, const int stride_w, const int col_h, const int col_w, Dtype* im){
	CUDA_KERNEL_LOOP(idx, n){
		Dtype val = 0;
		const int im_h = idx % height + pad_h;
		const int im_w = (idx / height) % width + pad_w;
		const int im_c = idx / height / width;
		const int w_start = (im_w < kernel_w) ? 0 : (im_w - kernel_w) / stride_w + 1;
		//	consider for condition1 and condition 3
		//	see more in im2col.cpp
		const int w_end = min(im_w / stride_w + 1, col_w);
		const int h_start = (im_h < kernel_h) ? 0 : (im_h - kernel_h) / stride_h + 1;
		const int h_end = min(im_h / stride_h + 1, col_h);

		for (int h = h_start; h < h_end; h++){
			for (int w = w_start; w < w_end; w++){
				int c = im_c * kernel_h * kernel_w
					+ (im_h - h * stride_h) * kernel_w + (im_w - w* stride_w);
				val += col[(c * col_h + h) * col_w + w];
			}
		}
		im[idx] = val;
	}
}

template<typename Dtype>
void col2im_gpu(const Dtype* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* im){
	const int col_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	const int col_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	const int n= (channels*height*width);
	col2im_gpu_kernel<Dtype> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(
		n, col, height, width, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, col_h, col_w, im);
	CUDA_POST_KERNEL_CHECK;
}

template void im2col_gpu<float>(const float* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* col);

template void im2col_gpu<double>(const double* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* col);

template void col2im_gpu<float>(const float* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* im);

template void col2im_gpu<double>(const double* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* im);