#include <cmath>
#include "layers/vision/roi_pooling_layer.hpp"

template <typename Dtype>
void ROIPoolingLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	ROIPoolingParameter roi_param = param.roi_pooling_param();
	CHECK_GT(roi_param.pooled_h(), 0);
	CHECK_GT(roi_param.pooled_w(), 0);
	pooling_height = roi_param.pooled_h();
	pooling_width = roi_param.pooled_w();
	spatial_scale = roi_param.spatial_scale();
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	//	ZF channels: 256
	channels = bottom[0]->channels();
	//	uncertain height/width while using different images
	height = bottom[0]->height();
	width = bottom[0]->width();
	//  mixed selective rois into network's batch_size axis
	//	e.g.
	//  conv5(bottom[0]) [1,256,height,width]
	//	rois(bottom[1])  [128,5]~[128,0/x1/y1/x2/y2]
	//	pooling_output(top[0]) [128,256,p_height,p_width]
	//	128 is the value specfied in CFG.TRAIN.BATCH_SIZE [default=128]
	top[0]->reshape(bottom[1]->num(), channels, pooling_height, pooling_width);
	max_idx.reshape(bottom[1]->num(), channels, pooling_height, pooling_width);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_rois = bottom[1]->cpu_data();
	int num_rois = bottom[1]->num();
	int im_batch_size = bottom[0]->num();
	int top_count = top[0]->count();
	Dtype* top_data = top[0]->mutable_cpu_data();
	dragon_set<Dtype>(top_count, Dtype(-FLT_MAX), top_data);
	int* argmax_data = max_idx.mutable_cpu_data();
	dragon_set<int>(top_count, -1, argmax_data);
	for (int n = 0; n < num_rois; n++){
		//	Faster-RCNN: im_batch_idx=0 (allow 1 image at a time)
		int im_batch_idx = bottom_rois[0];
		//	image_coordinate_system*spatial_scale = feature_map_coordinate system
		//	spatial_scale=2(conv1_stride)*2(pool1_stride)*2(conv2_stride)*2(pool2_stride)=16
		int x1 = round(bottom_rois[1] * spatial_scale);
		int y1 = round(bottom_rois[2] * spatial_scale);
		int x2 = round(bottom_rois[3] * spatial_scale);
		int y2 = round(bottom_rois[4] * spatial_scale);
		int roi_height = max(y2 - y1 + 1, 1);
		int roi_width = max(x2 - x1 + 1, 1);
		const Dtype unit_h = (Dtype)roi_height / (Dtype)pooling_height;
		const Dtype unit_w = (Dtype)roi_width / (Dtype)pooling_width;
		const Dtype* batch_data = bottom_data + bottom[0]->offset(im_batch_idx);
		// project roi_region into 6*6(default) pooling_region
		for (int c = 0; c < channels; c++){
			for (int ph = 0; ph < pooling_height; ph++){
				for (int pw = 0; pw < pooling_width; pw++){
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
					const int pooling_idx = ph*pooling_width + pw;
					if (is_empty){
						top_data[pooling_idx] = 0;
						argmax_data[pooling_idx] = -1;
					}
					for (int h = start_h; h < end_h; h++){
						for (int w = start_w; w < end_w; w++){
							const int idx = h*width + w;
							if (batch_data[idx]>top_data[pooling_idx]){
								top_data[pooling_idx] = batch_data[idx];
								argmax_data[pooling_idx] = idx;
							}
						}	//end w
					}	// end h
				}	// end pw
			}	// end ph

			//	offset image channels
			batch_data += bottom[0]->offset(0, 1);
			top_data += top[0]->offset(0, 1);
			argmax_data += max_idx.offset(0, 1);
		}	// end c
		// offset roi region
		bottom_rois += bottom[1]->offset(1);
	}	//end n
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ROIPoolingLayer);