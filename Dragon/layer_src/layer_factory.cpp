#include "layer_include/layer_factory.hpp"
#include "layer_include/vision_layers.hpp"
#include "layer_include/data_layers.hpp"
#include "layer_include/loss_layers.hpp"
#include "layer_include/neuron_layers.hpp"
#include "layer_include/common_layers.hpp"
REGISTER_LAYER_CLASS(Data);
REGISTER_LAYER_CLASS(AppData);
REGISTER_LAYER_CLASS(Convolution);
REGISTER_LAYER_CLASS(Pooling);
REGISTER_LAYER_CLASS(InnerProduct);
REGISTER_LAYER_CLASS(Accuracy);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);
REGISTER_LAYER_CLASS(ReLU);
REGISTER_LAYER_CLASS(Split);
REGISTER_LAYER_CLASS(BatchNorm);
REGISTER_LAYER_CLASS(Dropout);
