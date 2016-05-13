#include <cublas.h>
#include "utils/device.hpp"
#include "utils/math.hpp"


template<typename Dtype>
void dragon_gpu_copy(const int N, Dtype *dest, const Dtype *src){
	if (dest != src)
		CUDA_CHECK(cudaMemcpy(dest, src, N*sizeof(Dtype), cudaMemcpyDefault));
}

template void dragon_gpu_copy<int>(const int N, int *dest, const int *src);
template void dragon_gpu_copy<unsigned int>(const int N, unsigned int *dest, const unsigned int *src);
template void dragon_gpu_copy<float>(const int N, float *dest, const float *src);
template void dragon_gpu_copy<double>(const int N, double *dest, const double *src);

template <typename Dtype>
__global__ void set_kernel(const int N, const Dtype val, Dtype* x) {
	CUDA_KERNEL_LOOP(idx, N) {
		x[idx] = val;
	}
}

template <typename Dtype>
void dragon_gpu_set(const int N, const Dtype val, Dtype *x){
	if (val == 0){
		CUDA_CHECK(cudaMemset(x, 0, sizeof(Dtype)*N));
		return;
	}
	set_kernel<Dtype> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, val, x);
}

template void dragon_gpu_set<int>(const int N, const int val, int *x);
template void dragon_gpu_set<float>(const int N, const float val, float *x);
template void dragon_gpu_set<double>(const int N, const double val, double *x);

template<>
void dragon_gpu_gemm<float>(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
	const int M, const int N, const int K, const float alpha, const float* A, const float* B,
	const float beta, float *C){
	int lda = (transA == CblasNoTrans) ? K : M;
	int ldb = (transB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA =(transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB =(transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	// cublas run as col major and excute B*A but not A*B
	CUBLAS_CHECK(cublasSgemm_v2(Dragon::get_cublas_handle(), cuTransB, cuTransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template<>
void dragon_gpu_gemm<double>(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
	const int M, const int N, const int K, const double alpha, const double* A, const double* B,
	const double beta, double *C){
	int lda = (transA == CblasNoTrans) ? K : M;
	int ldb = (transB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	CUBLAS_CHECK(cublasDgemm_v2(Dragon::get_cublas_handle(), cuTransB, cuTransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template<>
void dragon_gpu_gemv<float>(const CBLAS_TRANSPOSE transA, const int M, const int N, const float alpha,
	const float* A, const float* x, const float beta, float* y){
	cublasOperation_t cuTransA =(transA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK(cublasSgemv_v2(Dragon::get_cublas_handle(), cuTransA, N, M, &alpha,A, N, x, 1, &beta, y, 1));
}

template<>
void dragon_gpu_gemv<double>(const CBLAS_TRANSPOSE transA, const int M, const int N, const double alpha,
	const double* A, const double* x, const double beta, double* y){
	cublasOperation_t cuTransA = (transA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	CUBLAS_CHECK(cublasDgemv_v2(Dragon::get_cublas_handle(), cuTransA, N, M, &alpha, A, N, x, 1, &beta, y, 1));
}

template<> void dragon_gpu_axpy<float>(int N, float alpha, const float *x, float *y){
	CUBLAS_CHECK(cublasSaxpy_v2(Dragon::get_cublas_handle(),N, &alpha, x, 1, y, 1));
}
template<> void dragon_gpu_axpy<double>(int N, double alpha, const double *x, double *y){
	CUBLAS_CHECK(cublasDaxpy_v2(Dragon::get_cublas_handle(), N, &alpha, x, 1, y, 1));
}

template <> void dragon_gpu_scal<float>(const int N, const float alpha, float* x){
	CUBLAS_CHECK(cublasSscal_v2(Dragon::get_cublas_handle(),N, &alpha, x, 1));
}
template <> void dragon_gpu_scal<double>(const int N, const double alpha, double* x){
	CUBLAS_CHECK(cublasDscal_v2(Dragon::get_cublas_handle(), N, &alpha, x, 1));
}

template<> void dragon_gpu_axpby<float>(int N, float alpha, const float *x, float beta, float *y){
	dragon_gpu_scal<float>(N, beta, y);
	dragon_gpu_axpy<float>(N, alpha, x, y);
}
template<> void dragon_gpu_axpby<double>(int N, double alpha, const double *x, double beta, double *y){
	dragon_gpu_scal<double>(N, beta, y);
	dragon_gpu_axpy<double>(N, alpha, x, y);
}

template<> float dragon_gpu_strided_dot<float>(const int N, const float* x, const int incx, const float* y, const int incy){
	float result;
	CUBLAS_CHECK(cublasSdot_v2(Dragon::get_cublas_handle(), N, x, incx, y, incy, &result));
	return result;
}
template<> double dragon_gpu_strided_dot<double>(const int N, const double* x, const int incx, const double* y, const int incy){
	double result;
	CUBLAS_CHECK(cublasDdot_v2(Dragon::get_cublas_handle(), N, x, incx, y, incy, &result));
	return result;
}

template <typename Dtype>
Dtype dragon_gpu_dot(const int N, const Dtype* x, const Dtype* y){
	return dragon_gpu_strided_dot<Dtype>(N, x, 1, y, 1);
}
template float dragon_gpu_dot<float>(const int N, const float* x, const float* y);
template double dragon_gpu_dot<double>(const int N, const double* x, const double* y);

template<> float dragon_gpu_asum<float>(int N, const float *x){
	return cublasSasum(N, x, 1);
}
template<> double dragon_gpu_asum<double>(int N, const double *x){
	return cublasDasum(N, x, 1);
}

template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx] = exp(a[idx]);
	}
}

template<> void dragon_gpu_exp<float>(int N, const float* x, float *y){
	exp_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, x, y);
}
template<> void dragon_gpu_exp<double>(int N, const double* x, double *y){
	exp_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, x, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx] = a[idx] / b[idx];
	}
}

template<> void dragon_gpu_div<float>(int N, const float* a, const float* b, float *y){
	div_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}
template<> void dragon_gpu_div<double>(int N, const double* a, const double* b, double *y){
	div_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx] = a[idx] * b[idx];
	}
}

template<> void dragon_gpu_mul<float>(int N, const float* a, const float* b, float *y){
	mul_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}
template<> void dragon_gpu_mul<double>(int N, const double* a, const double* b, double *y){
	mul_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a, const Dtype b, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx] = pow(a[idx], b);
	}
}

template<> void dragon_gpu_powx<float>(const int N, const float* a, const float b, float* y){
	powx_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}
template<> void dragon_gpu_powx<double>(const int N, const double* a, const double b, double* y){
	powx_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a, const Dtype *b, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx] = a[idx] + b[idx];
	}
}

template<> void dragon_gpu_add<float>(const int N, const float* a, const float *b, float* y){
	add_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}
template<> void dragon_gpu_add<double>(const int N, const double* a, const double *b, double* y){
	add_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a, const Dtype *b, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx] = a[idx] - b[idx];
	}
}

template<> void dragon_gpu_sub<float>(const int N, const float* a, const float *b, float* y){
	sub_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}
template<> void dragon_gpu_sub<double>(const int N, const double* a, const double *b, double* y){
	sub_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, a, b, y);
}

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, Dtype scalar, Dtype* y) {
	CUDA_KERNEL_LOOP(idx, n) {
		y[idx]+=scalar;
	}
}

template<> void dragon_gpu_add_scalar<float>(const int N, float scalar, float* y){
	add_scalar_kernel<float> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, scalar,y);
}
template<> void dragon_gpu_add_scalar<double>(const int N, double scalar, double* y){
	add_scalar_kernel<double> << <GET_BLOCKS(N), CUDA_NUM_THREADS >> >(N, scalar,y);
}

void dragon_gpu_rng_uniform(const int N, unsigned int* x){
	CURAND_CHECK(curandGenerate(Dragon::get_curand_generator(), x, N));
}

template <>
void dragon_gpu_scale<float>(const int N, const float alpha, const float *x,float* y) {
	CUBLAS_CHECK(cublasScopy_v2(Dragon::get_cublas_handle(), N, x, 1, y, 1));
	CUBLAS_CHECK(cublasSscal_v2(Dragon::get_cublas_handle(), N, &alpha, y, 1));
}

template <>
void dragon_gpu_scale<double>(const int N, const double alpha, const double *x, double* y) {
	CUBLAS_CHECK(cublasDcopy_v2(Dragon::get_cublas_handle(), N, x, 1, y, 1));
	CUBLAS_CHECK(cublasDscal_v2(Dragon::get_cublas_handle(), N, &alpha, y, 1));
}
