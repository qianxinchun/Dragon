#include <cmath>
#include "layers/vision/roi_pooling_layer.hpp"

template <typename Dtype>
__global__ void ROIPoolForward(const int n, const Dtype* bottom_data, const Dtype* bottom_rois,
	const Dtype spatial_scale, const int channels, const int height, const int width,
	const int pooling_height, const int pooling_width, Dtype* top_data, int* argmax_data){
	CUDA_KERNEL_LOOP(idx, n){
		int pw = idx % pooling_width;
		int ph = (idx / pooling_width) % pooling_height;
		int c = (idx / pooling_width / pooling_height) % channels;
		int n = idx / pooling_width / pooling_height / channels;
		// offset roi region
		bottom_rois += n * 5;
		int im_batch_idx = bottom_rois[0];
		int x1 = round(bottom_rois[1] * spatial_scale);
		int y1 = round(bottom_rois[2] * spatial_scale);
		int x2 = round(bottom_rois[3] * spatial_scale);
		int y2 = round(bottom_rois[4] * spatial_scale);
		int roi_height = max(y2 - y1 + 1, 1);
		int roi_width = max(x2 - x1 + 1, 1);
		const Dtype unit_h = (Dtype)roi_height / (Dtype)pooling_height;
		const Dtype unit_w = (Dtype)roi_width / (Dtype)pooling_width;

		//	compute base position
		int start_h = floor(unit_h*ph);
		int start_w = floor(unit_w*pw);
		int end_h = ceil(unit_h*(ph + 1));
		int end_w = ceil(unit_w*(pw + 1));
		//	compute offset position
		start_h = max(start_h + y1, 0);
		start_w = max(start_w + x1, 0);
		end_h = max(end_h + y1, 0);
		end_w = max(end_w + x1, 0);
		//	clip
		start_h = min(start_h, height);
		start_w = min(start_w, width);
		end_h = min(end_h, height);
		end_w = min(end_w, width);
		bool is_empty = (end_h == start_h) || (end_w == start_w);
		Dtype max_val = is_empty ? 0 : -FLT_MAX;
		int max_idx = -1;
		//	offset image channels
		bottom_data += (im_batch_idx*channels + c)*height*width;
		for (int h = start_h; h < end_h; h++){
			for (int w = start_w; w < end_w; w++){
				const int idx = h*width + w;
				if (bottom_data[idx]>max_val){
					max_val = bottom_data[idx];
					max_idx = idx;
				}
			}	//end w
		}	// end h
		top_data[idx] = max_val;
		argmax_data[idx] = max_idx;
	}
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* bottom_rois = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	int* argmax_data = max_idx.mutable_gpu_data();
	int count = top[0]->count();
	ROIPoolForward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, bottom_data, bottom_rois, spatial_scale, channels, height, width,
		pooling_height, pooling_width, top_data, argmax_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIPoolBackward(const int n, const Dtype* top_diff, const Dtype* bottom_rois,
	const int num_rois, const Dtype spatial_scale, const int channels, const int height, const int width,
	const int pooling_height, const int pooling_width, Dtype* bottom_diff, const int* argmax_data){
	CUDA_KERNEL_LOOP(idx, n){
		int w = idx % width;
		int h = (idx / width) % height;
		int c = (idx / width / height) % channels;
		int im_batch_idx = idx / width / height / channels;

		Dtype diff = 0;
		for (int n = 0; n < num_rois; n++){
			const Dtype* bottom_rois_off = bottom_rois + n * 5;
			int im_batch_idx2 = bottom_rois_off[0];
			//	ignore wrong im_batch_idx
			if (im_batch_idx != im_batch_idx2) continue;

			int x1 = round(bottom_rois_off[1] * spatial_scale);
			int y1 = round(bottom_rois_off[2] * spatial_scale);
			int x2 = round(bottom_rois_off[3] * spatial_scale);
			int y2 = round(bottom_rois_off[4] * spatial_scale);

			const bool is_in = (w >= x1&&w <= x2&&h >= y1&&h <= y2);
			//	ignore element out of feature map boundary directly
			//	or you can clip above 
			if (!is_in) continue;

			int roi_height = max(y2 - y1 + 1, 1);
			int roi_width = max(x2 - x1 + 1, 1);
			const Dtype unit_h = (Dtype)roi_height / (Dtype)pooling_height;
			const Dtype unit_w = (Dtype)roi_width / (Dtype)pooling_width;

			int start_ph = floor((h - y1) / unit_h);
			int start_pw = floor((w - x1) / unit_w);
			int end_ph = ceil((h + 1 - y1) / unit_h);
			int end_pw = ceil((w + 1 - x1) / unit_w);

			// cilp 
			start_ph = min(max(start_ph, 0), pooling_height);
			start_pw = min(max(start_pw, 0), pooling_width);
			end_ph = min(max(end_ph, 0), pooling_height);
			end_pw = min(max(end_pw, 0), pooling_width);

			int top_offset = (n*channels + c)*pooling_height*pooling_width;
			const Dtype* top_diff_off = top_diff + top_offset;
			const int* max_idx_off = argmax_data + top_offset;

			for (int ph = start_ph; ph < end_ph; ph++){
				for (int pw = start_pw; pw < end_pw; pw++){
					int pooling_idx = ph*pooling_width + pw;
					if (max_idx_off[pooling_idx] == (h*width + w)){
						diff += top_diff_off[pooling_idx];
					}
				}	//	end pw
			}	// end ph
		}	//	end n
		bottom_diff[idx] = diff;
	}
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (!data_need_bp[0]) return;
	const Dtype* bottom_rois = bottom[1]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	const int* argmax_data = max_idx.gpu_data();
	ROIPoolBackward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, top_diff, bottom_rois, bottom[1]->num(), spatial_scale, channels, height, width,
		pooling_height, pooling_width, bottom_diff, argmax_data);
	CUDA_POST_KERNEL_CHECK;

}


INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingLayer);