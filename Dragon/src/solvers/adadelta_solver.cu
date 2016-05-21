#include "solvers/gradient_solver.hpp"
#include <cmath>

template <typename Dtype>
__global__ void AdaDeltaUpdate(int n, Dtype* g, Dtype* h, Dtype* h2,
	Dtype momentum, Dtype eps, Dtype lr) {
	CUDA_KERNEL_LOOP(i, n) {
		float gi = g[i];
		float hi = h[i] = momentum * h[i] + (1 - momentum) * gi * gi;
		gi = gi * sqrt(h2[i] + eps) / sqrt(hi + eps);
		h2[i] = momentum * h2[i] + (1 - momentum) * gi * gi;
		g[i] = lr * gi;
	}
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::adadeltaUpdate(int n, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum, Dtype eps, Dtype lr) {
	AdaDeltaUpdate<Dtype> << <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, g, h, h2, momentum, eps, lr);
	CUDA_POST_KERNEL_CHECK;
}


template void AdaDeltaSolver<float>::adadeltaUpdate(int, float*, float*, float*, float, float, float);
template void AdaDeltaSolver<double>::adadeltaUpdate(int, double*, double*, double*, double, double, double);