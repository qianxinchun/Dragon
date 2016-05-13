#ifndef MATH_FUNCTIONS_HPP
#define MATH_FUNCTIONS_HPP

#include <openblas/cblas.h>
#include <cmath>
#include "../common.hpp"
#include "mkl_alternative.hpp"

template<typename Dtype>
void dragon_axpy(int N, Dtype alpha,const Dtype *x, Dtype *y);

template<typename Dtype>
void dragon_cpu_axpby(int N, Dtype alpha, const Dtype *x, Dtype beta,Dtype *y);

//	C=alpha*A*B+beta*C
template<typename Dtype>
void dragon_cpu_gemm(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
	const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B,
	const Dtype beta, Dtype *C);

//	y=alpha*A*x+beta*y
template<typename Dtype>
void dragon_cpu_gemv(const CBLAS_TRANSPOSE transA, const int M, const int N, const Dtype alpha,
	const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);

template<typename Dtype>
Dtype dragon_cpu_asum(int N, const Dtype *x);

template<typename Dtype>
void dragon_copy(const int N, Dtype *dest, const Dtype *src);

template <typename Dtype>
void dragon_set(const int N, const Dtype val, Dtype *x);

template <typename Dtype>
void dragon_rng_uniform(const int N, const Dtype lower, const Dtype upper, Dtype *x);

template <typename Dtype>
void dragon_rng_gaussian(const int N, const Dtype mu, const Dtype sigma, Dtype* x);

template <typename Dtype>
void dragon_rng_bernoulli(const int N, const Dtype p, unsigned int* x);

template <typename Dtype>
void dragon_exp(const int N, const Dtype* x, Dtype* y);

template <typename Dtype>
void dragon_div(const int N, const Dtype* a, const Dtype* b,Dtype* y);

template <typename Dtype>
void dragon_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
Dtype dragon_cpu_strided_dot(const int N, const Dtype* x, const int incx,const Dtype* y, const int incy);

template <typename Dtype>
Dtype dragon_cpu_dot(const int N, const Dtype* x, const Dtype* y);

template <typename Dtype>
void dragon_scal(const int N, const Dtype alpha, Dtype* x);

template <typename Dtype>
void dragon_scale(const int N, const Dtype alpha,const Dtype* x,Dtype* y);

template <typename Dtype>
void dragon_powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void dragon_add(const int N, const Dtype* a, const Dtype *b, Dtype* y);

template <typename Dtype>
void dragon_sub(const int N, const Dtype* a, const Dtype *b, Dtype* y);

template <typename Dtype>
void dragon_add_scalar(const int N, Dtype scalar,Dtype* y);

//	declare for GPU
#ifndef CPU_ONLY

template<typename Dtype>
void dragon_gpu_copy(const int N, Dtype *dest, const Dtype *src);

template<typename Dtype>
void dragon_gpu_set(const int N, const Dtype val, Dtype *x);

template<typename Dtype>
void dragon_gpu_axpy(int N, Dtype alpha, const Dtype *x, Dtype *y);

template<typename Dtype>
void dragon_gpu_axpby(int N, Dtype alpha, const Dtype *x, Dtype beta, Dtype *y);

template<typename Dtype>
Dtype dragon_gpu_asum(int N, const Dtype *x);

template<typename Dtype>
void dragon_gpu_gemm(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
	const int M, const int N, const int K, const Dtype alpha, const Dtype* A, const Dtype* B,
	const Dtype beta, Dtype *C);

template<typename Dtype>
void dragon_gpu_gemv(const CBLAS_TRANSPOSE transA, const int M, const int N, const Dtype alpha,
	const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);

template <typename Dtype>
void dragon_gpu_scal(const int N, const Dtype alpha, Dtype* x);

template <typename Dtype>
void dragon_gpu_scale(const int N, const Dtype alpha, const Dtype *x, Dtype* y);

template <typename Dtype>
Dtype dragon_gpu_strided_dot(const int N, const Dtype* x, const int incx, const Dtype* y, const int incy);

template <typename Dtype>
Dtype dragon_gpu_dot(const int N, const Dtype* x, const Dtype* y);

template <typename Dtype>
void dragon_gpu_exp(const int N, const Dtype* x, Dtype* y);

template <typename Dtype>
void dragon_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void dragon_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void dragon_gpu_powx(const int N, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void dragon_gpu_add(const int N, const Dtype* a, const Dtype *b, Dtype* y);

template <typename Dtype>
void dragon_gpu_sub(const int N, const Dtype* a, const Dtype *b, Dtype* y);

template <typename Dtype>
void dragon_gpu_add_scalar(const int N, Dtype scalar, Dtype* y);

void dragon_gpu_rng_uniform(const int N, unsigned int* x);

#endif






#endif