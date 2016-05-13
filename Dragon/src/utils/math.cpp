#include <boost/random.hpp>
#include "utils/math.hpp"
#include "utils/mkl_alternative.hpp"
#include "common.hpp"

template<typename Dtype>
void dragon_copy(const int N, Dtype *dest, const Dtype *src){
	if (dest != src) 
		memcpy(dest, src, sizeof(Dtype)*N);
}

template void dragon_copy<int>(const int N, int *dest, const int *src);
template void dragon_copy<unsigned int>(const int N, unsigned int *dest, const unsigned int *src);
template void dragon_copy<float>(const int N, float *dest, const float *src);
template void dragon_copy<double>(const int N, double *dest, const double *src);


template <typename Dtype>
void dragon_set(const int N, const Dtype val, Dtype *x){
	//memset run much faster than for(..) but only use with 0
	if (val == 0){
		memset(x, 0, sizeof(Dtype)*N);
		return;
	}
	for (int i = 0; i < N; i++) x[i] = val;
}

template void dragon_set<int>(const int N, const int val, int *x);
template void dragon_set<float>(const int N, const float val, float *x);
template void dragon_set<double>(const int N, const double val, double *x);

template<>
void dragon_cpu_gemm<float>(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
	const int M, const int N, const int K, const float alpha, const float* A, const float* B,
	const float beta, float *C){
	//	MAT[M,K] x MAT[K,N]
	//	in cblas document, ldx is the first dimension the mat x,
	//	first dimension with row major actually is col but not row
	//	that may be the default environment is Fortran using col major
	//	so,	the col is regard as the first dimension
	//	more refer to https://developer.apple.com/library/mac/documentation/Accelerate/Reference/BLAS_Ref/index.html#//apple_ref/doc/c_ref/CBLAS_TRANSPOSE
	int lda = (transA == CblasNoTrans) ? K : M;
	int ldb = (transB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template<>
void dragon_cpu_gemm<double>(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
	const int M, const int N, const int K, const double alpha, const double* A, const double* B,
	const double beta, double *C){
	int lda = (transA == CblasNoTrans) ? K : M;
	int ldb = (transB == CblasNoTrans) ? N : K;
	cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template<>
void dragon_cpu_gemv<float>(const CBLAS_TRANSPOSE transA, const int M, const int N, const float alpha,
	const float* A, const float* x, const float beta, float* y){
	int lda = (transA == CblasNoTrans) ? N : M;
	cblas_sgemv(CblasRowMajor, transA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void dragon_cpu_gemv<double>(const CBLAS_TRANSPOSE transA, const int M, const int N, const double alpha,
	const double* A, const double* x, const double beta, double* y){
	int lda = (transA == CblasNoTrans) ? N : M;
	cblas_dgemv(CblasRowMajor, transA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<typename Dtype>
void dragon_rng_uniform(const int N, const Dtype lower, const Dtype upper, Dtype *x){
	CHECK_GT(N, 0);
	CHECK(x);
	CHECK_LE(lower, upper);
	//	float distribution do not contain the upper bound
	//	nextafter will get the next float value after the specfic upper bound
	boost::uniform_real<Dtype> distribution(lower, std::nextafter(upper, std::numeric_limits<Dtype>::max()));
	//	we need variate_generator method which could use pointer to run the pseudo engine
	boost::variate_generator<rng_t*, boost::uniform_real<Dtype> > generator(Dragon::get_rng(), distribution);
	for (int i = 0; i < N; i++) x[i] = generator();
}

template void dragon_rng_uniform<float>(const int N, const float lower, const float upper, float *x);
template void dragon_rng_uniform<double>(const int N, const double lower, const double upper, double *x);

template<typename Dtype>
void dragon_rng_gaussian(const int N, const Dtype mu, const Dtype sigma,Dtype* x){
	CHECK_GT(N, 0);
	CHECK(x);
	//	sigma must greater than 0 in a normal distribution, or will divide "0" 
	CHECK_GT(sigma, 0);
	boost::normal_distribution<Dtype> distribution(mu, sigma);
	boost::variate_generator<rng_t*, boost::normal_distribution<Dtype> > generator(Dragon::get_rng(), distribution);
	for (int i = 0; i < N; i++) x[i] = generator();
}

template void dragon_rng_gaussian<float>(const int N, const float mu, const float sigma, float* x);
template void dragon_rng_gaussian<double>(const int N, const double mu, const double sigma, double* x);

template<typename Dtype>
void dragon_rng_bernoulli(const int N, const Dtype p, unsigned int* x){
	CHECK_GT(N, 0);
	CHECK(x);
	CHECK_GE(p, 0);
	CHECK_LE(p, 1);
	boost::bernoulli_distribution<Dtype> distribution(p);
	boost::variate_generator<rng_t*, boost::bernoulli_distribution<Dtype> > generator(Dragon::get_rng(), distribution);
	for (int i = 0; i < N; i++) x[i] = generator();
}

template void dragon_rng_bernoulli<float>(const int N, const float p, unsigned int* x);
template void dragon_rng_bernoulli<double>(const int N, const double p, unsigned int* x);


template<> void dragon_axpy<float>(int N,float alpha,const float *x,float *y){
	cblas_saxpy(N, alpha, x, 1, y, 1);
}
template<> void dragon_axpy<double>(int N, double alpha, const double *x, double *y){
	cblas_daxpy(N, alpha, x, 1, y, 1);
}

template<> void dragon_cpu_axpby<float>(int N, float alpha, const float *x, float beta,float *y){
	cblas_saxpby(N, alpha, x, 1, beta, y, 1);
}
template<> void dragon_cpu_axpby<double>(int N, double alpha, const double *x,double beta, double *y){
	cblas_daxpby(N, alpha, x, 1,beta, y, 1);
}

template<> float dragon_cpu_asum<float>(int N, const float *x){
	return cblas_sasum(N, x, 1);
}

template<> double dragon_cpu_asum<double>(int N, const double *x){
	return cblas_dasum(N, x, 1);
}

template<> void dragon_exp<float>(int N, const float* x, float *y){
	vsExp(N, x, y);
}

template<> void dragon_exp<double>(int N, const double* x, double *y){
	vdExp(N, x, y);
}

template<> void dragon_mul<float>(int N, const float* a, const float* b, float *y){
	vsMul(N, a, b, y);
}

template<> void dragon_mul<double>(int N, const double* a, const double* b, double *y){
	vdMul(N, a, b, y);
}

template<> void dragon_div<float>(int N, const float* a, const float* b,float *y){
	vsDiv(N, a, b, y);
}

template<> void dragon_div<double>(int N, const double* a, const double* b, double *y){
	vdDiv(N, a, b, y);
}

template<> float dragon_cpu_strided_dot<float>(const int N, const float* x, const int incx, const float* y, const int incy){
	return cblas_sdot(N, x, incx, y, incy);
}

template<> double dragon_cpu_strided_dot<double>(const int N, const double* x, const int incx,const double* y,const int incy){
	return cblas_ddot(N, x, incx, y, incy);
}

template <typename Dtype>
Dtype dragon_cpu_dot(const int N, const Dtype* x, const Dtype* y){
	return dragon_cpu_strided_dot<Dtype>(N, x, 1, y, 1);
}

template float dragon_cpu_dot<float>(const int N, const float* x, const float* y);
template double dragon_cpu_dot<double>(const int N, const double* x, const double* y);

template <> void dragon_scal<float>(const int N, const float alpha, float* x){
	cblas_sscal(N, alpha, x, 1);
}
template <> void dragon_scal<double>(const int N, const double alpha, double* x){
	cblas_dscal(N, alpha, x, 1);
}

template <> void dragon_scale<float>(const int N, const float alpha, const float* x,float* y){
	cblas_scopy(N, x, 1,y, 1);
	cblas_sscal(N, alpha, y, 1);
}
template <> void dragon_scale<double>(const int N, const double alpha, const double* x,double* y){
	cblas_dcopy(N, x, 1, y, 1);
	cblas_dscal(N, alpha, y, 1);
}

template <> void dragon_powx<float>(const int N, const float* a, const float b,float* y) {
	vsPowx(N, a, b, y);
}
template <> void dragon_powx<double>(const int N, const double* a, const double b,double* y) {
	vdPowx(N, a, b, y);
}

template <> void dragon_add<float>(const int N, const float* a, const float* b, float* y) {
	vsAdd(N, a, b, y);
}
template <>void dragon_add<double>(const int N, const double* a, const double* b, double* y) {
	vdAdd(N, a, b, y);
}

template <> void dragon_sub<float>(const int N, const float* a, const float* b, float* y) {
	vsSub(N, a, b, y);
}
template <>void dragon_sub<double>(const int N, const double* a, const double* b, double* y) {
	vdSub(N, a, b, y);
}

template <> void dragon_add_scalar<float>(const int N, float scalar, float* y) {
	for (int i = 0; i < N; i++) y[i] += scalar;
}
template <>void dragon_add_scalar<double>(const int N, double scalar, double* y) {
	for (int i = 0; i < N; i++) y[i] += scalar;
}