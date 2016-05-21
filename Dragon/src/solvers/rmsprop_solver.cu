#include "solvers/gradient_solver.hpp"
#include <cmath>

template <typename Dtype>
__global__ void RMSPropUpdate(int n, Dtype* g, Dtype* h,
	Dtype rms_decay, Dtype delta, Dtype lr) {
	CUDA_KERNEL_LOOP(i, n) {
		float gi = g[i];
		float hi = h[i] = rms_decay*h[i] + (1 - rms_decay)*gi*gi;
		g[i] = lr * g[i] / (sqrt(hi) + delta);
	}
}
template <typename Dtype>
void RMSPropSolver<Dtype>::rmspropUpdate(int n, Dtype* g, Dtype* h, Dtype momentum, Dtype eps, Dtype lr) {
	RMSPropUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
		<< <GET_BLOCKS(n), CUDA_NUM_THREADS >> >(n, g, h, momentum, eps, lr);
	CUDA_POST_KERNEL_CHECK;
}
template void RMSPropSolver<float>::rmspropUpdate(int, float*, float*, float, float,float);
template void RMSPropSolver<double>::rmspropUpdate(int, double*, double*, double, double,double);