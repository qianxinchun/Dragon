# ifndef IM2COL_HPP
# define IM2COL_HPP

template<typename Dtype>
void im2col_cpu(const Dtype* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* col);

template<typename Dtype>
void col2im_cpu(const Dtype* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* im);

template<typename Dtype>
void im2col_gpu(const Dtype* im, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* col);

template<typename Dtype>
void col2im_gpu(const Dtype* col, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* im);

# endif 