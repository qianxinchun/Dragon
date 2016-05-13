#include <vector>
#include "utils/im2col.hpp"
#include "utils/math.hpp"

template<typename Dtype>
void im2col_cpu(const Dtype* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* col){
	//	default using vaild convolution method
	//	which drops invaild pixels by formula (length-kernel+stride)
	//	condition_1(kernel>stride): named overlapping, formula drops the pixels at the end of row/col
	//	condition_2(kernel=stride): just split the length equally by stride(normally using in pooling operation)
	//	condition_3(kernel<stride): a special and uncommon case, skip some pixels in the middle of row/col
	//	condition_4(kernel=stride=1,pad=0): 1x1 is a special case, im2col do nothing,and we needn't do it to waste time
	const int col_h = (height + 2 * pad_h - kernel_h) / stride_h + 1; 
	const int col_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	//	for each element in kernel, create a row-map
	//	cause every for each input feature map, you use it's pixels col_h*col_w*kernel_h*kernel*w times
	//	so you need the size of col_c*col_h*col_w for storing corresponding pixels from input map
	//  col[] stores the im's pixel that x_th kernel_element's y_th computing need
	//	after yth computing, we will get y new pixels to combine a output map
	//	and the x_th element's 0th computing will have a x units offset on the im (x units should be splited in w/h axis)
	//	and we use padding for the im, the im_h and im_w will be out of range of im_shape
	//	check it and using "0" but not the im's pixel
	const int col_c = (channels*kernel_h*kernel_w);
	for (int c = 0; c < col_c; c++){
		int w_off = c % kernel_w;
		int h_off = (c / kernel_w) % kernel_h;
		int im_c = c / kernel_h / kernel_w;
		for (int h = 0; h < col_h; h++){
			for (int w = 0; w < col_w; w++){
				int im_h = h*stride_h - pad_h + h_off;
				int im_w = w*stride_w - pad_w + w_off;
				col[(c*col_h + h)*col_w + w] =
					(im_h >= 0 && im_w >= 0 && im_h < height&&im_w < width) ?
					im[(im_c*height + im_h)*width + im_w] : 0;
			}
		}
	}
}

template<typename Dtype>
void col2im_cpu(const Dtype* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* im){
	dragon_set(channels*height*width, Dtype(0),im);
	const int col_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	const int col_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	const int col_c = (channels*kernel_h*kernel_w);
	for (int c = 0; c < col_c; c++){
		int w_off = c % kernel_w;
		int h_off = (c / kernel_w) % kernel_h;
		int im_c = c / kernel_h / kernel_w;
		for (int h = 0; h < col_h; h++){
			for (int w = 0; w < col_w; w++){
				int im_h = h*stride_h - pad_h + h_off;
				int im_w = w*stride_w - pad_w + w_off;
				//	an im_pixel is cited by more col unit
				//	we sum them to compute diff
				if (im_h >= 0 && im_h < height&&im_w >= 0 && im_w < width){
					im[(im_c*height + im_h)*width + im_w] += col[(c*col_h + h)*col_w + w];
				}
			}
		}
	}
}


//	explicit instantiation for function
//	more info see http://bbs.csdn.net/topics/380250382

template void im2col_cpu<float>(const float* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* col);

template void im2col_cpu<double>(const double* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* col);

template void col2im_cpu<float>(const float* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* im);

template void col2im_cpu<double>(const double* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* im);


