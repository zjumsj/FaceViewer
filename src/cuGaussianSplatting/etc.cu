
#include "etc.h"

#include <stdio.h>
//#include <exception>
#include <stdexcept>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <cub/cub.cuh>

#include "sh_rot.cuh"

// Gaussian splatting's SH definition is different from
// wiki https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
// Enable this flag to do transformation
#define NEG_SH_SIGN

// DEBUG
//#include "../cnpy/cnpy.h"


#define MY_SURF_BOUNDARY_MODE cudaBoundaryModeTrap

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float length4(float* x) {
	return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
}

__forceinline__ __device__ int padding_shared_memory(int x, int n_pad = 1) {
	return x + (x / 32 * n_pad);
}

__device__ void quaternion_raw_multiply(
	float* c,
	const float* a,
	const float* b
) {
	c[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
	c[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
	c[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
	c[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

__device__ void quaternion_to_matrix(float* mat, const float* quat) {
	float r = quat[0];
	float i = quat[1];
	float j = quat[2];
	float k = quat[3];
	float two_s = 2.f / (r * r + i * i + j * j + k * k);
	mat[0] = 1.f - two_s * (j * j + k * k);
	mat[1] = two_s * (i * j - k * r);
	mat[2] = two_s * (i * k + j * r);
	mat[3] = two_s * (i * j + k * r);
	mat[4] = 1.f - two_s * (i * i + k * k);
	mat[5] = two_s * (j * k - i * r);
	mat[6] = two_s * (i * k - j * r);
	mat[7] = two_s * (j * k + i * r);
	mat[8] = 1.f - two_s * (i * i + j * j);
}

__device__ void matrix_to_quaternion(float * outq, const float* mat) {	
	float q_abs[4];
	q_abs[0] = 1 + mat[0] + mat[4] + mat[8];
	q_abs[1] = 1 + mat[0] - mat[4] - mat[8];
	q_abs[2] = 1 - mat[0] + mat[4] - mat[8];
	q_abs[3] = 1 - mat[0] - mat[4] + mat[8];
#pragma unroll
	for (int i = 0; i < 4; i++) {
		if (q_abs[i] > 0) q_abs[i] = sqrt(q_abs[i]);
		else q_abs[i] = 0;
	}
	float div;
	int argmax_id = 0;
	float max_v = q_abs[0];
	if (q_abs[1] > max_v) {	max_v = q_abs[1]; argmax_id = 1;}
	if (q_abs[2] > max_v) { max_v = q_abs[2]; argmax_id = 2;}
	if (q_abs[3] > max_v) { max_v = q_abs[3]; argmax_id = 3;}
	if (argmax_id == 0) {		
		outq[0] = q_abs[0] * q_abs[0];
		outq[1] = mat[7] - mat[5];
		outq[2] = mat[2] - mat[6];
		outq[3] = mat[3] - mat[1];
	}
	else if (argmax_id == 1) {
		outq[0] = mat[7] - mat[5];
		outq[1] = q_abs[1] * q_abs[1];
		outq[2] = mat[3] + mat[1];
		outq[3] = mat[2] + mat[6];
	}
	else if (argmax_id == 2) {
		outq[0] = mat[2] - mat[6];
		outq[1] = mat[3] + mat[1];
		outq[2] = q_abs[2] * q_abs[2];
		outq[3] = mat[5] + mat[7];
	}else{
		outq[0] = mat[3] - mat[1];
		outq[1] = mat[6] + mat[2];
		outq[2] = mat[7] + mat[5];
		outq[3] = q_abs[3] * q_abs[3];
	}
	div = q_abs[argmax_id]; // div 0 should never happen ?
	//if (div < 0.1f) div = 0.1f;
	div *= 2;
	outq[0] /= div;
	outq[1] /= div;
	outq[2] /= div;
	outq[3] /= div;
}

__device__ void inv_of_mat3x3(const float* src, float* tar) {

	float det = src[0] * (src[4] * src[8] - src[7] * src[5]) -
		src[1] * (src[3] * src[8] - src[5] * src[6]) +
		src[2] * (src[3] * src[7] - src[4] * src[6]);
	float invdet = 1.f / det;
	tar[0] = (src[4] * src[8] - src[7] * src[5]) * invdet;
	tar[1] = (src[2] * src[7] - src[1] * src[8]) * invdet;
	tar[2] = (src[1] * src[5] - src[2] * src[4]) * invdet;
	tar[3] = (src[5] * src[6] - src[3] * src[8]) * invdet;
	tar[4] = (src[0] * src[8] - src[2] * src[6]) * invdet;
	tar[5] = (src[3] * src[2] - src[0] * src[5]) * invdet;
	tar[6] = (src[3] * src[7] - src[6] * src[4]) * invdet;
	tar[7] = (src[6] * src[1] - src[0] * src[7]) * invdet;
	tar[8] = (src[0] * src[4] - src[3] * src[1]) * invdet;
}

__device__ void transpose_mat3x3(const float* src, float* tar) {
	float tmp;
	tar[0] = src[0]; tar[1] = src[3]; tar[2] = src[6];
	tar[3] = src[1]; tar[4] = src[4]; tar[5] = src[7];
	tar[6] = src[2]; tar[7] = src[5]; tar[8] = src[8];
}

__device__ void normalize_transform(const float* src, float* tar, int N = 5) {

	float* mat = tar;
	float mat_T[9];
	float mat_Tinv[9];
	for (int j = 0; j < 9; j++) {
		mat[j] = src[j];
	}

	for (int i = 0; i < N; i++) {
		//float mat_next[9];
		transpose_mat3x3(mat, mat_T);
		inv_of_mat3x3(mat_T, mat_Tinv);
		for (int j = 0; j < 9; j++) {
			mat[j] = 0.5f * (mat[j] + mat_Tinv[j]);
		}
	}
}

__device__ void colormap_jet(float v, float* color){
	if (v < 0.0) v = 0.0;
	if (v > 1.0) v = 1.0;

	color[0] = 0.f;
	color[1] = 0.f;
	color[2] = 0.f;
	// G
	if (v < 0.125f) {
		color[1] = 0.f;
	}
	else if (v < 0.375f) {
		color[1] = (v - 0.125f) / (0.375f - 0.125f);
	}
	else if (v < 0.64f) {
		color[1] = 1.f;
	}
	else if (v < 0.91f) {
		color[1] = 1.f - (v - 0.64f) / (0.91f - 0.64f);
	}
	// R
	if (v < 0.35f) {
		color[0] = 0.f;
	}
	else if (v < 0.66f) {
		color[0] = (v - 0.35f) / (0.66f - 0.35f);
	}
	else if (v < 0.888f) {
		color[0] = 1.f;
	}
	else {
		color[0] = 0.5f + 4.464285714285714f * (1.f - v);
	}
	// B
	if (v < 0.112f) {
		color[2] = 0.5f + 4.464285714285714f * v;
	}
	else if (v < 0.34f) {
		color[2] = 1.f;
	}
	else if (v < 0.65f) {
		color[2] = 1.f - (v - 0.34f) / (0.65f - 0.34f);
	}
}

//////////////////////////////

template<int THREADS>
__global__ void CopyRotKernel(
	int64_t n_elements,
	const float* __restrict__ src,
	float* __restrict__ tar,
	bool use_activation
) {
	__shared__ float sdata[THREADS * 4 + (THREADS * 4) / 32];
#pragma unroll
	for (int i = 0; i < 4; i++) {
		int loc_offset = i * THREADS + threadIdx.x;
		int64_t i_elem = blockIdx.x * (THREADS * 4) + loc_offset;
		if (i_elem < n_elements * 4) {
			sdata[padding_shared_memory(loc_offset)] = src[i_elem];
		}
	}
	__syncthreads();
	int64_t i_elem = blockIdx.x * THREADS + threadIdx.x;
	float loc_rot[4];
	if (i_elem < n_elements) {
		loc_rot[0] = sdata[padding_shared_memory(threadIdx.x * 4 + 0)];
		loc_rot[1] = sdata[padding_shared_memory(threadIdx.x * 4 + 1)];
		loc_rot[2] = sdata[padding_shared_memory(threadIdx.x * 4 + 2)];
		loc_rot[3] = sdata[padding_shared_memory(threadIdx.x * 4 + 3)];
		if (use_activation) {
			float length = length4(loc_rot);
			loc_rot[0] /= length;
			loc_rot[1] /= length;
			loc_rot[2] /= length;
			loc_rot[3] /= length;
		}
	}
	__syncthreads();
	if (i_elem < n_elements) {
		sdata[padding_shared_memory(threadIdx.x * 4 + 0)] = loc_rot[0];
		sdata[padding_shared_memory(threadIdx.x * 4 + 1)] = loc_rot[1];
		sdata[padding_shared_memory(threadIdx.x * 4 + 2)] = loc_rot[2];
		sdata[padding_shared_memory(threadIdx.x * 4 + 3)] = loc_rot[3];
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 4; i++) {
		int loc_offset = i * THREADS + threadIdx.x;
		int64_t i_elem = blockIdx.x * (THREADS * 4) + loc_offset;
		if (i_elem < n_elements * 4) {
			tar[i_elem] = sdata[padding_shared_memory(loc_offset)];
		}
	}
}

template<int P>
__global__ void CopyScaleKernel(
	int64_t n_elements, 
	const float * __restrict__ src,
	float * __restrict__ tar,
	bool use_activation
) {
	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t i_elem = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		if (i_elem < n_elements) {
			float v = src[i_elem];
			if (use_activation) {
				v = __expf(v);
			}
			tar[i_elem] = v;
		}
	}
}

template<int P>
__global__ void CopyOpacityKernel(
	int64_t n_elements,
	const float* __restrict__ src,
	float* __restrict__ tar,
	bool use_activation
) {
	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t i_elem = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		if (i_elem < n_elements) {
			float v = src[i_elem];
			if (use_activation) {
				v = sigmoid(v);
			}
			tar[i_elem] = v;
		}
	}
}


__global__ void TransferPosKernel(
	int64_t n_elements,
	const float * __restrict__ mat4x3,
	const float * __restrict__ src,
	float * __restrict__ tar
) {
	__shared__ float smat[12];
	if (threadIdx.x < 12) {
		smat[threadIdx.x] = mat4x3[threadIdx.x];
	}
	__syncthreads();
	float loc_mat[12];
	for (int i = 0; i < 12; i++) {
		loc_mat[i] = smat[i];
	}
	int64_t i_elem = blockIdx.x * blockDim.x + threadIdx.x;
	if (i_elem < n_elements) {
		float src_pos[3];
		float tar_pos[3];
		src_pos[0] = src[3 * i_elem + 0];
		src_pos[1] = src[3 * i_elem + 1];
		src_pos[2] = src[3 * i_elem + 2];
		tar_pos[0] = loc_mat[0] * src_pos[0] + loc_mat[1] * src_pos[1] + loc_mat[2] * src_pos[2] + loc_mat[3];
		tar_pos[1] = loc_mat[4] * src_pos[0] + loc_mat[5] * src_pos[1] + loc_mat[6] * src_pos[2] + loc_mat[7];
		tar_pos[2] = loc_mat[8] * src_pos[0] + loc_mat[9] * src_pos[1] + loc_mat[10] * src_pos[2] + loc_mat[11];
		tar[3 * i_elem + 0] = tar_pos[0];
		tar[3 * i_elem + 1] = tar_pos[1];
		tar[3 * i_elem + 2] = tar_pos[2];		
	}
}

__global__ void TransferRotActivationKernel(
	int64_t n_elements,
	const float* __restrict__ qrot4,
	const float* __restrict__ src,
	float* __restrict__ tar
) {
	__shared__ float srot[4];
	if (threadIdx.x < 4) {
		srot[threadIdx.x] = qrot4[threadIdx.x];
	}
	__syncthreads();
	float loc_rot[4];
	for (int i = 0; i < 4; i++) {
		loc_rot[i] = srot[i];
	}
	int64_t i_elem = blockIdx.x * blockDim.x + threadIdx.x;
	if (i_elem < n_elements) {
		float v[4];
		v[0] = src[i_elem * 4 + 0];
		v[1] = src[i_elem * 4 + 1];
		v[2] = src[i_elem * 4 + 2];
		v[3] = src[i_elem * 4 + 3];
		float length = length4(v);
		v[0] = v[0] / length;
		v[1] = v[1] / length;
		v[2] = v[2] / length;
		v[3] = v[3] / length;
		// do rot
		float tmpv[4];
		tmpv[0] = v[0]; tmpv[1] = v[1]; tmpv[2] = v[2]; tmpv[3] = v[3];
		quaternion_raw_multiply(v, loc_rot, tmpv);
		tar[i_elem * 4 + 0] = v[0];
		tar[i_elem * 4 + 1] = v[1];
		tar[i_elem * 4 + 2] = v[2];
		tar[i_elem * 4 + 3] = v[3];
	}
}

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32 * 3) TransferSHKernel(
	int64_t n_elements,
	const float* __restrict__ mat3x3,
	const float* __restrict__ src,
	float* __restrict__ tar
) {
	__shared__ float s_shs[32 * 3 * 16];
	__shared__ float s_mat[9];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = src[i_elem];
		}
	}
	if (threadIdx.x < 9) {
		s_mat[threadIdx.x] = mat3x3[threadIdx.x];
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if (i_elem < n_elements) {
		float loc_sh[16];
		float loc_mat[9];
		// copy sh to local
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh[i] = -loc_sh[i];
		}
#endif
		// copy transform matrix to local
		for (int i = 0; i < 9; i++) {
			loc_mat[i] = s_mat[i];
		}
		float sh1[3][3];
		float sh2[5][5];
		float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);
		loc_sh_out[0] = loc_sh[0];

		loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[0][1] * loc_sh[2] + sh1[0][2] * loc_sh[3];
		loc_sh_out[2] = sh1[1][0] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[1][2] * loc_sh[3];
		loc_sh_out[3] = sh1[2][0] * loc_sh[1] + sh1[2][1] * loc_sh[2] + sh1[2][2] * loc_sh[3];
#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[i][j] * loc_sh[4 + j];
			}
		}
#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[i][j] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh_out[i] = -loc_sh_out[i];
		}
#endif
	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			tar[i_elem] = s_shs[loc_offset];
		}
	}
	
}

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32 * 3) TransferSHKernelQuat(
	int64_t n_elements,
	const float* __restrict__ quatT, // 4xP
	const float* __restrict__ src, // Px16x3
	float * __restrict__ tar // Px16x3
) {
	__shared__ float s_shs[32 * 3 * 16];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = src[i_elem];
		}
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if (i_elem < n_elements) {
		float loc_sh[16];
		float loc_quat[4];
		float loc_mat[9];
		// copy sh to local
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh[i] = -loc_sh[i];
		}
#endif
		// copy transform matrix to local
		//for (int i = 0; i < 9; i++) {
		//	loc_mat[i] = s_mat[i];
		//}
		loc_quat[0] = quatT[n_elements * 0 + i_elem];
		loc_quat[1] = quatT[n_elements * 1 + i_elem];
		loc_quat[2] = quatT[n_elements * 2 + i_elem];
		loc_quat[3] = quatT[n_elements * 3 + i_elem];
		quaternion_to_matrix(loc_mat, loc_quat);

		float sh1[3][3];
		float sh2[5][5];
		float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);
		loc_sh_out[0] = loc_sh[0];

		loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[0][1] * loc_sh[2] + sh1[0][2] * loc_sh[3];
		loc_sh_out[2] = sh1[1][0] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[1][2] * loc_sh[3];
		loc_sh_out[3] = sh1[2][0] * loc_sh[1] + sh1[2][1] * loc_sh[2] + sh1[2][2] * loc_sh[3];
#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[i][j] * loc_sh[4 + j];
			}
		}
#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[i][j] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh_out[i] = -loc_sh_out[i];
		}
#endif
	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			tar[i_elem] = s_shs[loc_offset];
		}
	}
}

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32 * 3) TransferSHKernelQuat_Inplace(
	int64_t n_elements,
	const float* __restrict__ quatT, // 4xP
	float* __restrict__ in_out // Px16x3
) {
	__shared__ float s_shs[32 * 3 * 16];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = in_out[i_elem];
		}
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if (i_elem < n_elements) {
		float loc_sh[16];
		float loc_quat[4];
		float loc_mat[9];
		// copy sh to local
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh[i] = -loc_sh[i];
		}
#endif
		// copy transform matrix to local
		//for (int i = 0; i < 9; i++) {
		//	loc_mat[i] = s_mat[i];
		//}
		loc_quat[0] = quatT[n_elements * 0 + i_elem];
		loc_quat[1] = quatT[n_elements * 1 + i_elem];
		loc_quat[2] = quatT[n_elements * 2 + i_elem];
		loc_quat[3] = quatT[n_elements * 3 + i_elem];
		quaternion_to_matrix(loc_mat, loc_quat);

		float sh1[3][3];
		float sh2[5][5];
		float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);
		loc_sh_out[0] = loc_sh[0];

		loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[0][1] * loc_sh[2] + sh1[0][2] * loc_sh[3];
		loc_sh_out[2] = sh1[1][0] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[1][2] * loc_sh[3];
		loc_sh_out[3] = sh1[2][0] * loc_sh[1] + sh1[2][1] * loc_sh[2] + sh1[2][2] * loc_sh[3];
#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[i][j] * loc_sh[4 + j];
			}
		}
#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[i][j] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh_out[i] = -loc_sh_out[i];
		}
#endif
	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			in_out[i_elem] = s_shs[loc_offset];
		}
	}
}

// FUFace edition
template<int SN>
__global__ void CompositeBasisPosKernel(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	const float * __restrict__ params, // expr(51)
	const float * __restrict__ pos_mean, // {Px3}
	const float * __restrict__ pos_basis, // {Px3}xK expr
	float * __restrict__ pos_tar // {Px3}
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float v[3];
		v[0] = pos_mean[i_elem * 3 + 0];
		v[1] = pos_mean[i_elem * 3 + 1];
		v[2] = pos_mean[i_elem * 3 + 2];
		for (int ii = 0; ii < p_n; ii++) { // expr basis
			float k = tile[ii];
			int real_i = ii + p_offset;
			v[0] += pos_basis[(i_elem * 3 + 0) * p_total + real_i] * k;
			v[1] += pos_basis[(i_elem * 3 + 1) * p_total + real_i] * k;
			v[2] += pos_basis[(i_elem * 3 + 2) * p_total + real_i] * k;
		}
		pos_tar[i_elem * 3 + 0] = v[0];
		pos_tar[i_elem * 3 + 1] = v[1];
		pos_tar[i_elem * 3 + 2] = v[2];
	}
}

// FLAME edition
template<int SN>
__global__ void CompositeBasisPosKernel(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	int p_offset2, int p_n2,
	const float* __restrict__ params, // expr(100) + eyelid(2) + flatten_node(36)
	const float* __restrict__ pos_mean, // {Px3}
	const float* __restrict__ pos_basis, // {Px3}xK expr
	const float* __restrict__ pos_basis2, // 36xPx3 pos
	float* __restrict__ pos_tar // {Px3}
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n + p_n2) {
		const float* l_ptr;
		if (threadIdx.x < p_n) {
			l_ptr = &params[p_offset + threadIdx.x];
		}
		else {
			l_ptr = &params[p_offset2 + (threadIdx.x - p_n)];
		}
		tile[threadIdx.x] = *l_ptr;
	}
	__syncthreads();
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float v[3];
		v[0] = pos_mean[i_elem * 3 + 0];
		v[1] = pos_mean[i_elem * 3 + 1];
		v[2] = pos_mean[i_elem * 3 + 2];
		for (int ii = 0; ii < p_n; ii++) { // expr basis
			float k = tile[ii];
			int real_i = ii + p_offset;
			v[0] += pos_basis[(i_elem * 3 + 0) * p_total + real_i] * k;
			v[1] += pos_basis[(i_elem * 3 + 1) * p_total + real_i] * k;
			v[2] += pos_basis[(i_elem * 3 + 2) * p_total + real_i] * k;
		}
		for (int ii = 0; ii < p_n2; ii++) { // pose basis
			float k = tile[p_n + ii];
			v[0] += pos_basis2[(ii * n_elements + i_elem) * 3 + 0] * k;
			v[1] += pos_basis2[(ii * n_elements + i_elem) * 3 + 1] * k;
			v[2] += pos_basis2[(ii * n_elements + i_elem) * 3 + 2] * k;
		}
		pos_tar[i_elem * 3 + 0] = v[0];
		pos_tar[i_elem * 3 + 1] = v[1];
		pos_tar[i_elem * 3 + 2] = v[2];
	}
}


template<int SN>
__global__ void CompositeRotKernel_Inplace(
	int64_t n_elements, 
	const float* __restrict__ params,
	const float* __restrict__ quatT, // 4xP
	float * __restrict__ rot_inout // {Px4}
) {
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float v[4];
		v[0] = rot_inout[i_elem * 4 + 0];
		v[1] = rot_inout[i_elem * 4 + 1];
		v[2] = rot_inout[i_elem * 4 + 2];
		v[3] = rot_inout[i_elem * 4 + 3];
		float loc_quatT[4];
		loc_quatT[0] = quatT[0 * n_elements + i_elem];
		loc_quatT[1] = quatT[1 * n_elements + i_elem];
		loc_quatT[2] = quatT[2 * n_elements + i_elem];
		loc_quatT[3] = quatT[3 * n_elements + i_elem];
		float tmpv[4];
		tmpv[0] = v[0]; tmpv[1] = v[1]; tmpv[2] = v[2]; tmpv[3] = v[3];
		quaternion_raw_multiply(v, loc_quatT, tmpv);
		rot_inout[i_elem * 4 + 0] = v[0];
		rot_inout[i_elem * 4 + 1] = v[1];
		rot_inout[i_elem * 4 + 2] = v[2];
		rot_inout[i_elem * 4 + 3] = v[3];
	}
}

template<int SN>
__global__ void CompositeBasisRotKernel3(
	int64_t n_elements, int p_offset, int p_n, int p_total, bool enable_rot,
	const float* __restrict__ params,
	const float* __restrict__ rot_mean, // Px4
	const float* __restrict__ rot_basis, // Px4xK 
	const float* __restrict__ ptr_DR, // 4xP
	float* __restrict__ rot_tar // {Px4}
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float v[4];
		v[0] = rot_mean[i_elem * 4 + 0];
		v[1] = rot_mean[i_elem * 4 + 1];
		v[2] = rot_mean[i_elem * 4 + 2];
		v[3] = rot_mean[i_elem * 4 + 3];

		if (enable_rot) {
			float deform_rot[4];			
			deform_rot[0] = ptr_DR[0 * n_elements + i_elem];
			deform_rot[1] = ptr_DR[1 * n_elements + i_elem];
			deform_rot[2] = ptr_DR[2 * n_elements + i_elem];
			deform_rot[3] = ptr_DR[3 * n_elements + i_elem];

			float tmpv[4];
			tmpv[0] = v[0]; tmpv[1] = v[1]; tmpv[2] = v[2]; tmpv[3] = v[3];
			quaternion_raw_multiply(v, deform_rot, tmpv);
		}
		//////////
		for (int ii = 0; ii < p_n; ii++) {
			float k = tile[ii];
			int real_i = ii + p_offset;
			v[0] += rot_basis[(i_elem * 4 + 0) * p_total + real_i] * k;
			v[1] += rot_basis[(i_elem * 4 + 1) * p_total + real_i] * k;
			v[2] += rot_basis[(i_elem * 4 + 2) * p_total + real_i] * k;
			v[3] += rot_basis[(i_elem * 4 + 3) * p_total + real_i] * k;
		}
		// do normalization
		float length = length4(v);
		rot_tar[i_elem * 4 + 0] = v[0] / length;
		rot_tar[i_elem * 4 + 1] = v[1] / length;
		rot_tar[i_elem * 4 + 2] = v[2] / length;
		rot_tar[i_elem * 4 + 3] = v[3] / length;
	}
}

template<int THREADS>
__global__ void RotPosAndAddEyelidKernel_Inplace(
	int64_t n_elements, int p_offset,
	const float* __restrict__ params, // expr(100) + eyelid(2) + flatten_node(36)
	const float* __restrict__ T, // 12xP
	const float* __restrict__ eyelid, // {P}x3x2
	float* __restrict__ pos_inout
) {
	__shared__ float spos[THREADS * 3];
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * THREADS + threadIdx.x;
		int64_t i_elem = blockIdx.x * (THREADS * 3) + loc_offset;
		if (i_elem < n_elements * 3) {
			spos[loc_offset] = pos_inout[i_elem];
		}
	}
	__syncthreads();
	int64_t i_elem = blockIdx.x * THREADS + threadIdx.x;
	float pos[3];
	float pos_out[3];
	if (i_elem < n_elements) {
		pos[0] = spos[threadIdx.x * 3 + 0];
		pos[1] = spos[threadIdx.x * 3 + 1];
		pos[2] = spos[threadIdx.x * 3 + 2];
		//pos_out[0] = pos[0];
		//pos_out[1] = pos[1];
		//pos_out[2] = pos[2];
		float loc_T[12];
		for (int j = 0; j < 12; j++) {
			loc_T[j] = T[j * n_elements + i_elem];
		}
		pos_out[0] = loc_T[0] * pos[0] + loc_T[1] * pos[1] + loc_T[2] * pos[2] + loc_T[3];
		pos_out[1] = loc_T[4] * pos[0] + loc_T[5] * pos[1] + loc_T[6] * pos[2] + loc_T[7];
		pos_out[2] = loc_T[8] * pos[0] + loc_T[9] * pos[1] + loc_T[10] * pos[2] + loc_T[11];
		if (eyelid) {
			float k1 = params[p_offset];
			float k2 = params[p_offset + 1];
			//float k1 = 1.f;
			//float k2 = 0.f;
			pos_out[0] += eyelid[i_elem * 6 + 0] * k1 + eyelid[i_elem * 6 + 1] * k2;
			pos_out[1] += eyelid[i_elem * 6 + 2] * k1 + eyelid[i_elem * 6 + 3] * k2;
			pos_out[2] += eyelid[i_elem * 6 + 4] * k1 + eyelid[i_elem * 6 + 5] * k2;
		}
	}
	__syncthreads();
	if (i_elem < n_elements) {
		spos[threadIdx.x * 3 + 0] = pos_out[0];
		spos[threadIdx.x * 3 + 1] = pos_out[1];
		spos[threadIdx.x * 3 + 2] = pos_out[2];
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * THREADS + threadIdx.x;
		int64_t i_elem = blockIdx.x * (THREADS * 3) + loc_offset;
		if (i_elem < n_elements * 3) {
			pos_inout[i_elem] = spos[loc_offset];
		}
	}	
}

template<int SN>
__global__ void CompositeDeformRot(
	int64_t n_elements, int p_offset, int p_n, int p_total, bool do_norm,
	const float* __restrict__ params,
	const float* __restrict__ rot_d, // Px4xK
	float * __restrict__ ptr_DR // 4xP
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float deform_rot[4] = { 1.f,0.f,0.f,0.f };
		for (int ii = 0; ii < p_n; ii++) {
			float k = tile[ii];
			int real_i = ii + p_offset;
			deform_rot[0] += rot_d[(i_elem * 4 + 0) * p_total + real_i] * k;
			deform_rot[1] += rot_d[(i_elem * 4 + 1) * p_total + real_i] * k;
			deform_rot[2] += rot_d[(i_elem * 4 + 2) * p_total + real_i] * k;
			deform_rot[3] += rot_d[(i_elem * 4 + 3) * p_total + real_i] * k;
		}
		if (do_norm) {
			float length = length4(deform_rot);
			deform_rot[0] = deform_rot[0] / length;
			deform_rot[1] = deform_rot[1] / length;
			deform_rot[2] = deform_rot[2] / length;
			deform_rot[3] = deform_rot[3] / length;
		}
		ptr_DR[0 * n_elements + i_elem] = deform_rot[0];
		ptr_DR[1 * n_elements + i_elem] = deform_rot[1];
		ptr_DR[2 * n_elements + i_elem] = deform_rot[2];
		ptr_DR[3 * n_elements + i_elem] = deform_rot[3];
	}
}

template<int THREADS>
__global__ void CompositeT(
	int64_t n_elements,
	const float* __restrict__ W, // {P}x5
	const float* __restrict__ node_transfer, // 5x4x4
	float* __restrict__ T, // 12xP
	float* __restrict__ quatT // 4xP
) {
	__shared__ float sweight[THREADS * 5];
	__shared__ float snode[5 * 4 * 4];
#pragma unroll
	for (int i = 0; i < 5; i++) {
		int loc_offset = i * THREADS + threadIdx.x;
		int64_t i_elem = blockIdx.x * (THREADS * 5) + loc_offset;
		if (i_elem < n_elements * 5) {
			sweight[loc_offset] = W[i_elem];
		}
	}
	if (threadIdx.x < 5 * 4 * 4) {
		snode[threadIdx.x] = node_transfer[threadIdx.x];
	}
	__syncthreads();
	int64_t i_elem = blockIdx.x * THREADS + threadIdx.x;
	if (i_elem < n_elements) {
		
		float loc_mat[12];
		float loc_rot[4];

		for (int j = 0; j < 12; j++)
			loc_mat[j] = 0.f;
		
		for (int i_node = 0; i_node < 5; i_node++) {
			float w = sweight[threadIdx.x * 5 + i_node];
			for (int j = 0; j < 12; j++) {
				loc_mat[j] += snode[i_node * 16 + j] * w;
			}
		}

		float rot_mat[9];
		float norm_rot_mat[9];
		rot_mat[0] = loc_mat[0]; rot_mat[1] = loc_mat[1]; rot_mat[2] = loc_mat[2];
		rot_mat[3] = loc_mat[4]; rot_mat[4] = loc_mat[5]; rot_mat[5] = loc_mat[6];
		rot_mat[6] = loc_mat[8]; rot_mat[7] = loc_mat[9]; rot_mat[8] = loc_mat[10];
		normalize_transform(rot_mat, norm_rot_mat);
		matrix_to_quaternion(loc_rot, norm_rot_mat);
		// write back
		for (int j = 0; j < 12; j++) {
			T[j * n_elements + i_elem] = loc_mat[j];
		}
		for (int j = 0; j < 4; j++) {
			quatT[j * n_elements + i_elem] = loc_rot[j];
		}
	}
}



////////////////////////////////////////////

// threads = 32 * 3
// shared memory for sh = 32 * 3 * 16
// shared memory for matrix = 32 * 9
__global__ void __launch_bounds__(32*3) CompositeBasisRotSHKernel(
	int64_t n_elements, //int p_offset, int p_n, int p_total,
	const float* __restrict__ faceNR,
	const float* __restrict__ shs_in, // {P}xLx3
	float* __restrict__ shs_out // {P}xLx3
) {
	__shared__ float s_shs[32 * 3 * 16];
	__shared__ float s_mat[32 * 9];
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			s_shs[loc_offset] = shs_in[i_elem];
		}
	}
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 3) + loc_offset;
		if (i_elem < n_elements * 9) {
			s_mat[loc_offset] = faceNR[i_elem];
		}
	}
	__syncthreads();
	int64_t thread_idx = blockIdx.x * 96 + threadIdx.x;
	int64_t i_elem = thread_idx / 3;
	int li_elem = threadIdx.x / 3;
	int channel = threadIdx.x % 3;
	float loc_sh_out[16];
	if (i_elem < n_elements) {		
		float loc_sh[16];
		float loc_mat[9];		
#pragma unroll
		for (int i = 0; i < 16; i++) {
			loc_sh[i] = s_shs[li_elem * 48 + i * 3 + channel];
		}
#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh[i] = -loc_sh[i];
		}
#endif
//#pragma unroll
//		for (int i = 0; i < 9; i++) {
//			loc_mat[i] = s_mat[li_elem * 9 + i];
//		}
// 
		// Eigen Matrix in col major !!
		loc_mat[0] = s_mat[li_elem * 9 + 0]; loc_mat[1] = s_mat[li_elem * 9 + 3]; loc_mat[2] = s_mat[li_elem * 9 + 6];
		loc_mat[3] = s_mat[li_elem * 9 + 1]; loc_mat[4] = s_mat[li_elem * 9 + 4]; loc_mat[5] = s_mat[li_elem * 9 + 7];
		loc_mat[6] = s_mat[li_elem * 9 + 2]; loc_mat[7] = s_mat[li_elem * 9 + 5]; loc_mat[8] = s_mat[li_elem * 9 + 8];

		float sh1[3][3];
		float sh2[5][5];
		float sh3[7][7];
		Construct_SH_Rotation_Matrix(loc_mat, sh1, sh2, sh3);

		/*for (int i = 0; i < 16; i++) {
			loc_sh_out[i] = loc_sh[i];
		}*/
		
		loc_sh_out[0] = loc_sh[0];

		loc_sh_out[1] = sh1[0][0] * loc_sh[1] + sh1[0][1] * loc_sh[2] + sh1[0][2] * loc_sh[3];
		loc_sh_out[2] = sh1[1][0] * loc_sh[1] + sh1[1][1] * loc_sh[2] + sh1[1][2] * loc_sh[3];
		loc_sh_out[3] = sh1[2][0] * loc_sh[1] + sh1[2][1] * loc_sh[2] + sh1[2][2] * loc_sh[3];

#pragma unroll
		for (int i = 0; i < 5; i++) {
			loc_sh_out[4 + i] = 0;
#pragma unroll
			for (int j = 0; j < 5; j++) {
				loc_sh_out[4 + i] += sh2[i][j] * loc_sh[4 + j];
			}
		}

#pragma unroll
		for (int i = 0; i < 7; i++) {
			loc_sh_out[9 + i] = 0;
#pragma unroll
			for (int j = 0; j < 7; j++) {
				loc_sh_out[9 + i] += sh3[i][j] * loc_sh[9 + j];
			}
		}

#ifdef NEG_SH_SIGN
#pragma unroll
		for (int i = 0; i < 16; i++) {
			if (i % 2) loc_sh_out[i] = -loc_sh_out[i];
		}
#endif

	}
	//// write back, first to shared memory, then to global memory
	__syncthreads();
	if (i_elem < n_elements) {
#pragma unroll
		for (int i = 0; i < 16; i++) {
			s_shs[li_elem * 48 + i * 3 + channel] = loc_sh_out[i];
		}
	}
	__syncthreads();
#pragma unroll
	for (int i = 0; i < 16; i++) {
		int loc_offset = i * 96 + threadIdx.x;
		int64_t i_elem = blockIdx.x * (96 * 16) + loc_offset;
		if (i_elem < n_elements * 48) {
			shs_out[i_elem] = s_shs[loc_offset];
		}
	}
}


/////////////////////////////// ACC KERNEL

template<int SN>
__global__ void __launch_bounds__(64) CompositeBasisSHKernelAccN32T64_Inplace(
	int64_t n_elements,
	const float * __restrict__ params,
	float * shs, // {PxLx3}
	const float * __restrict__ shs_basis // {PxLx3}xK
) {
	__shared__ float tile[SN + 32 * SN];
	const int n_load_pass = (SN + 32 * SN + 63) / 64;
	const float* ptr;

	for (int i = 0; i < n_load_pass; i++) {
		int loc_offset = i * 64 + threadIdx.x;
		bool load = true;
		if (loc_offset < SN) {
			ptr = params + loc_offset;
		}
		else {
			int64_t i_elem = (blockIdx.x * 32) * SN + (loc_offset - SN);
			ptr = shs_basis + i_elem;
			if (i_elem >= n_elements * SN || (loc_offset - SN) >= 32 * SN)
				load = false;
		}
		if (load) {
			tile[loc_offset] = *ptr;
		}
	}
	__syncthreads();
	// split 32 elements -> 16(warp0) + 16(warp1) to reduce shared memory bank conflicts
	if (threadIdx.x % 32 < 16) {
		int thread_idx = (threadIdx.x / 32) * 16 + (threadIdx.x % 32);
		int64_t i_elem = blockIdx.x * 32 + thread_idx;
		if (i_elem < n_elements) {
			float v = shs[i_elem];
			for (int ii = 0; ii < SN; ii++) {
				float k = tile[ii];
				float bt = tile[SN + (thread_idx * SN + ii)];
				v += bt * k;
			}
			shs[i_elem] = v;
		}
	}
}


template<int SN>
__global__ void __launch_bounds__(128) CompositeBasisSHKernelAccN128T128_Inplace(
	int64_t n_elements,
	const float* __restrict__ params,
	float* shs, //{PxLx3}
	const float* __restrict__ shs_basis // {PxLx3}xK
) {
	__shared__ float tile[SN + 128 * SN];// TODO: May meet bank conflict, 50% efficiency
	const int n_load_pass = (SN + 128 * SN + 127) / 128;
	const float* ptr;

	for (int i = 0; i < n_load_pass; i++) {
		int loc_offset = i * 128 + threadIdx.x;
		bool load = true;
		if (loc_offset < SN) {
			ptr = params + loc_offset;
		}
		else {
			int64_t i_elem = (blockIdx.x * 128) * SN + (loc_offset - SN);
			ptr = shs_basis + i_elem;
			if (i_elem >= n_elements * SN || (loc_offset - SN) >= 128 * SN)
				load = false;
		}
		if (load) {
			tile[loc_offset] = *ptr;
		}
	}
	__syncthreads();

	int64_t i_elem = blockIdx.x * 128 + threadIdx.x;
	if (i_elem < n_elements) {
		float v = shs[i_elem];
		for (int ii = 0; ii < SN; ii++) {
			float k = tile[ii];
			float bt = tile[SN + (threadIdx.x * SN + ii)];
			v += bt * k;
		}
		shs[i_elem] = v;
	}
}

template<int SN>
__global__ void __launch_bounds__(64) CompositeBasisSHKernelAccN64T64_Inplace(
	int64_t n_elements,
	const float * __restrict__ params,
	float* shs, //{PxLx3}
	const float * __restrict__ shs_basis // {PxLx3}xK
) {
	__shared__ float tile[SN + 64 * SN];// TODO: May meet bank conflict, 50% efficiency
	const int n_load_pass = (SN + 64 * SN + 63) / 64;
	const float* ptr;

	for (int i = 0; i < n_load_pass; i++) {
		int loc_offset = i * 64 + threadIdx.x;
		bool load = true;
		if (loc_offset < SN) {
			ptr = params + loc_offset;
		}
		else {
			int64_t i_elem = (blockIdx.x * 64) * SN + (loc_offset - SN);
			ptr = shs_basis + i_elem;
			if (i_elem >= n_elements * SN || (loc_offset - SN) >= 64 * SN)
				load = false;
		}
		if (load) {
			tile[loc_offset] = *ptr;
		}
	}
	__syncthreads();
		
	int64_t i_elem = blockIdx.x * 64 + threadIdx.x;
	if (i_elem < n_elements) {
		float v = shs[i_elem];
		for (int ii = 0; ii < SN; ii++) {
			float k = tile[ii];
			float bt = tile[SN + (threadIdx.x * SN + ii)];
			v += bt * k;
		}
		shs[i_elem] = v;
	}	
}

template<int SN>
__global__ void __launch_bounds__(64) CompositeBasisSHKernelAccN64T64(
	int64_t n_elements,
	const float* __restrict__ params,
	const float* __restrict__ shs_mean, // {PxLx3}
	const float* __restrict__ shs_basis, // {PxLx3}xK
	float* __restrict__ shs_tar  // {PxLx3}
) {
	__shared__ float tile[SN + 64 * SN];// TODO: May meet bank conflict, 50% efficiency
	const int n_load_pass = (SN + 64 * SN + 63) / 64;
	const float* ptr;
	
	for (int i = 0; i < n_load_pass; i++) {
		int loc_offset = i * 64 + threadIdx.x;
		bool load = true;
		if (loc_offset < SN) {
			ptr = params + loc_offset;
		}
		else {
			int64_t i_elem = (blockIdx.x * 64) * SN + (loc_offset - SN);
			ptr = shs_basis + i_elem;
			if (i_elem >= n_elements * SN || (loc_offset - SN) >= 64 * SN)
				load = false;
		}
		if (load) {
			tile[loc_offset] = *ptr;
		}
	}
	__syncthreads();

	int64_t i_elem = blockIdx.x * 64 + threadIdx.x;
	if (i_elem < n_elements) {
		float v = shs_mean[i_elem];
		for (int ii = 0; ii < SN; ii++) {
			float k = tile[ii];
			float bt = tile[SN + (threadIdx.x * SN + ii)];
			v += bt * k;
		}
		shs_tar[i_elem] = v;
	}
}

template<int SN>
__global__ void __launch_bounds__(256) CompositeBasisSHKernelAccN32T256_Inplace(
	int64_t n_elements, 
	const float* __restrict__ params,
	float* shs, // {PxLx3}
	const float* __restrict__ shs_basis // {PxLx3}xK
) {

	__shared__ float tile[SN + 32 * SN];// TODO: May meet bank conflict, 50% efficiency
	const int n_load_pass = (SN + 32*SN + 255) / 256;
	const float* ptr;
	
	for (int i = 0; i < n_load_pass; i++) {
		int loc_offset = i * 256 + threadIdx.x;
		bool load = true;
		if (loc_offset < SN) {
			ptr = params + loc_offset;
		}
		else {
			int64_t i_elem = (blockIdx.x * 32) * SN + (loc_offset - SN);
			ptr = shs_basis + i_elem;
			if (i_elem >= n_elements * SN || (loc_offset - SN) >= 32 * SN)
				load = false;
		}
		if (load) {
			tile[loc_offset] = *ptr;
		}
	}
	__syncthreads();

	if (threadIdx.x < 32) {
		int64_t i_elem = blockIdx.x * 32 + threadIdx.x;
		if (i_elem < n_elements) {
			float v = shs[i_elem];
			for (int ii = 0; ii < SN; ii++) {
				float k = tile[ii];
				float bt = tile[SN + (threadIdx.x * SN + ii)];
				v += bt * k;
			}
			shs[i_elem] = v;
		}	
	}

}


/////////////////////////////// END ACC KERNEL

template<int P, int SN>
__global__ void CompositeBasisSHKernel_Inplace(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	const float* __restrict__ params,
	float *  shs, // {PxLx3}
	const float* __restrict__ shs_basis // {PxLx3}xK
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t i_elem = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		if (i_elem < n_elements) {
			float v = shs[i_elem];
			for (int ii = 0; ii < p_n; ii++) {
				float k = tile[ii];
				int real_i = ii + p_offset;
				v += shs_basis[i_elem * p_total + real_i] * k;
			}
			shs[i_elem] = v;
			//shs_tar[i_elem] = v;
		}
	}
}

template<int P,int SN>
__global__ void CompositeBasisSHKernel(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	const float* __restrict__ params,
	const float* __restrict__ shs_mean, // {PxLx3}
	const float* __restrict__ shs_basis, // {PxLx3}xK
	float * __restrict__ shs_tar  // {PxLx3}
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t i_elem = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		if (i_elem < n_elements) {
			float v = shs_mean[i_elem];
			for (int ii = 0; ii < p_n; ii++) {
				float k = tile[ii];
				int real_i = ii + p_offset;
				v += shs_basis[i_elem * p_total + real_i] * k;
			}
			shs_tar[i_elem] = v;
		}
	}
}

template<int P, int SN>
__global__ void CompositeBasisScaleKernel(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	const float* __restrict__ params,
	const float* __restrict__ scale_mean, // {Px3}
	const float* __restrict__ scale_basis, // {Px3}xK
	float* __restrict__ scale_tar  // {Px3}
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t i_elem = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		if (i_elem < n_elements) {
			float v = scale_mean[i_elem];
			for (int ii = 0; ii < p_n; ii++) {
				float k = tile[ii];
				int real_i = ii + p_offset;
				v += scale_basis[i_elem * p_total + real_i] * k;
			}
			v = __expf(v);
			scale_tar[i_elem] = v;
			//scale_tar[i_elem] = 0.001f;
		}
	}	
}


template<int SN>
__global__ void CompositeBasisRotKernel2(
	int64_t n_elements, int p_offset, int p_n, int p_total, bool enable_rot,
	const float* __restrict__ params,
	const float* __restrict__ faceNR,
	const float* __restrict__ rot_mean, // {P}x4
	const float* __restrict__ rot_basis, // {P}x4xK
	float* __restrict__ rot_tar // {P}x4
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float v[4];
		v[0] = rot_mean[i_elem * 4 + 0];
		v[1] = rot_mean[i_elem * 4 + 1];
		v[2] = rot_mean[i_elem * 4 + 2];
		v[3] = rot_mean[i_elem * 4 + 3];
		// do normalize
		float length = length4(v);
		v[0] = v[0] / length;
		v[1] = v[1] / length;
		v[2] = v[2] / length;
		v[3] = v[3] / length;
		if (enable_rot) {
			float loc_faceNR[9];
			//for (int i = 0; i < 9; i++) {
			//	loc_faceNR[i] = faceNR[9 * i_elem + i];
			//}
			
			// Eigen Matrix is col major !!
			loc_faceNR[0] = faceNR[i_elem * 9 + 0]; loc_faceNR[1] = faceNR[i_elem * 9 + 3]; loc_faceNR[2] = faceNR[i_elem * 9 + 6];
			loc_faceNR[3] = faceNR[i_elem * 9 + 1]; loc_faceNR[4] = faceNR[i_elem * 9 + 4]; loc_faceNR[5] = faceNR[i_elem * 9 + 7];
			loc_faceNR[6] = faceNR[i_elem * 9 + 2]; loc_faceNR[7] = faceNR[i_elem * 9 + 5]; loc_faceNR[8] = faceNR[i_elem * 9 + 8];
			
			float q_faceNR[4];
			float tmpv[4];
			matrix_to_quaternion(q_faceNR, loc_faceNR);
			tmpv[0] = v[0]; tmpv[1] = v[1]; tmpv[2] = v[2]; tmpv[3] = v[3];
			quaternion_raw_multiply(v, q_faceNR, tmpv);
		}
		//////////
		for (int ii = 0; ii < p_n; ii++) {
			float k = tile[ii];
			int real_i = ii + p_offset;
			v[0] += rot_basis[(i_elem * 4 + 0) * p_total + real_i] * k;
			v[1] += rot_basis[(i_elem * 4 + 1) * p_total + real_i] * k;
			v[2] += rot_basis[(i_elem * 4 + 2) * p_total + real_i] * k;
			v[3] += rot_basis[(i_elem * 4 + 3) * p_total + real_i] * k;
		}
		//// normalize quaternion
		length = length4(v);
		rot_tar[i_elem * 4 + 0] = v[0] / length;
		rot_tar[i_elem * 4 + 1] = v[1] / length;
		rot_tar[i_elem * 4 + 2] = v[2] / length;
		rot_tar[i_elem * 4 + 3] = v[3] / length;
	}
}




template<int SN>
__global__ void CompositeBasisRotKernel(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	const float* __restrict__ params,
	const float* __restrict__ rot_mean, // {P}x4
	const float* __restrict__ rot_basis, // {P}x4xK
	float* __restrict__ rot_tar  // {P}x4
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	int64_t i_elem = (int64_t)blockIdx.x * blockDim.x + (int64_t)threadIdx.x;
	if (i_elem < n_elements) {
		float v[4];
		v[0] = rot_mean[i_elem * 4 + 0];
		v[1] = rot_mean[i_elem * 4 + 1];
		v[2] = rot_mean[i_elem * 4 + 2];
		v[3] = rot_mean[i_elem * 4 + 3];
		for (int ii = 0; ii < p_n; ii++) {
			float k = tile[ii];
			int real_i = ii + p_offset;
			v[0] += rot_basis[(i_elem * 4 + 0) * p_total + real_i] * k;
			v[1] += rot_basis[(i_elem * 4 + 1) * p_total + real_i] * k;
			v[2] += rot_basis[(i_elem * 4 + 2) * p_total + real_i] * k;
			v[3] += rot_basis[(i_elem * 4 + 3) * p_total + real_i] * k;
		}
		//// normalize quaternion
		float length = length4(v);
		rot_tar[i_elem * 4 + 0] = v[0] / length;
		rot_tar[i_elem * 4 + 1] = v[1] / length;
		rot_tar[i_elem * 4 + 2] = v[2] / length;
		rot_tar[i_elem * 4 + 3] = v[3] / length;
	}	
}

template<int P, int SN>
__global__ void CompositeBasisOpacityKernel(
	int64_t n_elements, int p_offset, int p_n, int p_total,
	const float* __restrict__ params,
	const float* __restrict__ opacity_mean, // {Px1}
	const float* __restrict__ opacity_basis, // {Px1}xK
	float* __restrict__ opacity_tar  // {Px1}
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();
	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t i_elem = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		if (i_elem < n_elements) {
			float v = opacity_mean[i_elem];
			for (int ii = 0; ii < p_n; ii++) {
				float k = tile[ii];
				int real_i = ii + p_offset;
				v += opacity_basis[i_elem * p_total + real_i] * k;
			}
			v = sigmoid(v);
			opacity_tar[i_elem] = v;
			//opacity_tar[i_elem] = 0.5f;
		}
	}
}

///////////////////////////

template<int P, int SN>
__global__ void BlendTexKernel(
	int height, int width, int p_offset, int p_n, int p_total,
	float * __restrict__ basis, // HxWx3xk
	float * __restrict__ means, // HxWx3
	float * __restrict__ params,
	float scale, bool bgr, cudaSurfaceObject_t surfobj
) {
	__shared__ float tile[SN];
	if (threadIdx.x < p_n) {
		tile[threadIdx.x] = params[p_offset + threadIdx.x];
	}
	__syncthreads();

	for (int i_loop = 0; i_loop < P; i_loop++) {
		int64_t id = ((int64_t)blockIdx.x * P + i_loop) * blockDim.x + (int64_t)threadIdx.x;
		int i = (int)(id % width);
		int j = (int)(id / width);

		if (j < height) {
			float4 ans;
			ans.x = ans.y = ans.z = 0;
			if (means) {
				if (bgr) {
					ans.z = means[id * 3 + 0] * scale;
					ans.y = means[id * 3 + 1] * scale;
					ans.x = means[id * 3 + 2] * scale;
				}
				else {
					ans.x = means[id * 3 + 0] * scale;
					ans.y = means[id * 3 + 1] * scale;
					ans.z = means[id * 3 + 2] * scale;
				}
			}
			for (int ii = 0; ii < p_n; ii++) {
				float k = tile[ii];
				int real_i = ii + p_offset;
				if (bgr) {
					ans.z += k * basis[(id * 3 + 0) * p_total + real_i] * scale;
					ans.y += k * basis[(id * 3 + 1) * p_total + real_i] * scale;
					ans.x += k * basis[(id * 3 + 2) * p_total + real_i] * scale;
				}
				else {
					ans.x += k * basis[(id * 3 + 0) * p_total + real_i] * scale;
					ans.y += k * basis[(id * 3 + 1) * p_total + real_i] * scale;
					ans.z += k * basis[(id * 3 + 2) * p_total + real_i] * scale;
				}
			}
			//ans.x = tile[0];
			//ans.y = tile[1];
			//ans.z = tile[2];
			ans.w = 1;
			//ans.x = 1; ans.y = 0.5; ans.z = 0.5; ans.w = 1;

			surf2Dwrite(ans, surfobj, (int)sizeof(float4) * i, j, MY_SURF_BOUNDARY_MODE);
		}
	}
}

__global__ void fillColorKernel(
	float* __restrict__ gpu_buffer,
	int n_points, float R, float G, float B
) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < n_points * 3) {
		float v = (thread_id % 3 == 0) ? R : ((thread_id % 3 == 1) ? G : B);
		gpu_buffer[thread_id] = v;
	}
}

__global__ void colorDepthKernel(
	float* __restrict__ canvas,
	float k, float b, bool inverse,
	int n_pixels
) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	float color[3];

	if (thread_id < n_pixels) {
		float v = canvas[0 * n_pixels + thread_id];
		float sv = v * k + b;
		if (inverse)
			sv = 1.f - sv;
		colormap_jet(sv, color);
		if (v == 0.f) { // set back to black
			color[0] = 0.f;
			color[1] = 0.f;
			color[2] = 0.f;
		}
		canvas[0 * n_pixels + thread_id] = color[0];
		canvas[1 * n_pixels + thread_id] = color[1];
		canvas[2 * n_pixels + thread_id] = color[2];
	}
}

template<int SN>
__global__ void computeDepthKernel(
	float* __restrict__ gpu_buffer,
	const float* __restrict__ pos,
	const float* __restrict__ modelview,
	int n_points
) {
	__shared__ float tile[SN * 3];
	int thread_id = blockIdx.x * SN + threadIdx.x;
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_id = (blockIdx.x * 3 + i) * SN + threadIdx.x;
		if (loc_id < n_points * 3) {
			tile[SN * i + threadIdx.x] = pos[loc_id];
		}
	}
	__syncthreads();

	float outpos[3];
	if (thread_id < n_points) {
		float posx, posy, posz;
		posx = tile[threadIdx.x * 3 + 0];
		posy = tile[threadIdx.x * 3 + 1];
		posz = tile[threadIdx.x * 3 + 2];		
		//outpos[0] = modelview[0] * posx + modelview[4] * posy + modelview[8] * posz + modelview[12];
		//outpos[1] = modelview[1] * posx + modelview[5] * posy + modelview[9] * posz + modelview[13];
		outpos[2] = modelview[2] * posx + modelview[6] * posy + modelview[10] * posz + modelview[14];
		//gpu_buffer[thread_id] = outpos[2]; // +z
	}

	__syncthreads(); // make sure the first task is done

	if (thread_id < n_points) {
		tile[threadIdx.x * 3 + 0] = outpos[2];
		tile[threadIdx.x * 3 + 1] = outpos[2];
		tile[threadIdx.x * 3 + 2] = outpos[2];
	}

	__syncthreads();

#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_id = (blockIdx.x * 3 + i) * SN + threadIdx.x;
		if (loc_id < n_points * 3) {
			gpu_buffer[loc_id] = tile[SN * i + threadIdx.x];
		}
	}
}

template<int SN>
__global__ void visAnisotropyKernel(
	float* __restrict__ fill_color,
	const float* __restrict__ scale,
	int n_points, int max_rate
) {
	__shared__ float tile[SN * 3];
	int thread_id = blockIdx.x * SN + threadIdx.x;
#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_id = (blockIdx.x * 3 + i) * SN + threadIdx.x;
		if (loc_id < n_points * 3) {
			tile[SN * i + threadIdx.x] = scale[loc_id];
		}
	}
	__syncthreads();

	float color = 0.f;

	if (thread_id < n_points) {
		float sx, sy, sz;
		sx = tile[threadIdx.x * 3 + 0];
		sy = tile[threadIdx.x * 3 + 1];
		sz = tile[threadIdx.x * 3 + 2];
		float min_s = sx;
		float max_s = sx;
		if (sy < min_s) min_s = sy;
		if (sy > max_s) max_s = sy;
		if (sz < min_s) min_s = sz;
		if (sz > max_s) max_s = sz;
		float rate = (max_s + 1e-12) / (min_s + 1e-12);
		color = (rate - 1.f) / (max_rate - 1.f);
	}

	__syncthreads(); // make sure the first task is done

	if (thread_id < n_points) {
		tile[threadIdx.x * 3 + 0] = color;
		tile[threadIdx.x * 3 + 1] = color;
		tile[threadIdx.x * 3 + 2] = color;
	}

	__syncthreads();

#pragma unroll
	for (int i = 0; i < 3; i++) {
		int loc_id = (blockIdx.x * 3 + i) * SN + threadIdx.x;
		if (loc_id < n_points * 3) {
			fill_color[loc_id] = tile[SN * i + threadIdx.x];
		}
	}

}
	

namespace cuGaussianSplatting {
	
	extern struct cudaGraphicsResource * resources[];

	inline int nBlock(int n, int blockSize) {
		return (n + blockSize - 1) / blockSize;
	}

	inline int64_t nBlock(int64_t n, int64_t blockSize) {
		return (n + blockSize - 1) / blockSize;
	}

	void BlendTex(
		int height, int width, int p_offset, int p_n, int p_total,
		void* basis, void* means, void* params, int register_idx, 
		float scale, bool bgr
	) {
		cudaError_t cudaStatus = cudaSuccess;
		cudaArray* arr_ptr1;

		cudaStatus = cudaGraphicsMapResources(1, &resources[register_idx], 0);// 96 us
		if (cudaStatus != cudaSuccess) throw std::runtime_error("cudaGraphicsMapResources fail!\n");

		cudaStatus = cudaGraphicsSubResourceGetMappedArray(&arr_ptr1, resources[register_idx], 0, 0);
		if (cudaStatus != cudaSuccess) throw std::runtime_error("cudaGraphicsSubResourceGetMappedArray fail!\n");

		{
			cudaSurfaceObject_t surfObj = 0;
			struct cudaResourceDesc resDesc;
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = arr_ptr1;
			cudaStatus = cudaCreateSurfaceObject(&surfObj, &resDesc);
			if (cudaStatus != cudaSuccess) throw std::runtime_error("cudaCreateSurfaceObject fail!\n");
			const unsigned int loop = 4;
			const unsigned int block = 512;
			unsigned int grid = nBlock(width * height, block * loop);

			if (p_n <= 256) {
				// input.stride, input.point_offset, color_offset,
				BlendTexKernel<loop,256> << <grid, block >> > (
					height,width,p_offset, p_n, p_total,
					(float*)basis,(float*)means,(float*)params,
					scale, bgr, surfObj
				);
			}
			else {
				throw std::runtime_error("no proper kernel!");
			}
			//cudaStatus = cudaDeviceSynchronize();
			//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail12!\n");

			//cudaStatus = cudaGetLastError();
			//if (cudaStatus != cudaSuccess) throw std::runtime_error("BlendTexKernel fail!\n");
			
			cudaStatus = cudaDestroySurfaceObject(surfObj);
			if (cudaStatus != cudaSuccess) throw std::runtime_error("cudaDestroySurfaceObject fail!\n");
		}

		cudaStatus = cudaGraphicsUnmapResources(1, &resources[register_idx], 0);
		if (cudaStatus != cudaSuccess) throw std::runtime_error("cudaGraphicsUnmapResources fail!\n");
	}

	void CheckCudaError(int i);

	void fillColorCore(float* gpu_buffer, int n_points, const float* cpu_color3) {
		cudaError_t cudaStatus = cudaSuccess;		
		const unsigned int block = 512;
		unsigned int grid = nBlock(n_points * 3,block);
		fillColorKernel << <grid, block >> > (gpu_buffer, n_points, cpu_color3[0], cpu_color3[1], cpu_color3[2]);		
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail13!\n");
	}	

	void fillColorWithDepthCore(
		float * gpu_buffer, const float * pos, int n_points, const float * modelview
	) {
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int block = 256;
		unsigned int grid = nBlock(n_points, block);
		computeDepthKernel<block> <<<grid, block >>> (gpu_buffer, pos, modelview, n_points);
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail13!\n");
		
		// Determine temporary device storage requirements
		//void* d_temp_storage = NULL;
		//size_t   temp_storage_bytes = 0;
		//cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, gpu_buffer, minmaxbuffer, n_points, min_op, 1e12);

		//// Allocate temporary storage
		//cudaMalloc(&d_temp_storage, temp_storage_bytes);

		//// Run reduction (min)
		//cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, gpu_buffer, minmaxbuffer, n_points, min_op, 1e12);
		//// Run reduction (max)
		//cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, gpu_buffer, minmaxbuffer + 1, n_points, max_op, -1e12);
		//cudaFree(d_temp_storage);		
	}

	void ColorDepthCore(
		float* canvas, int width, int height, float k, float b, bool inverse
	) {
		int n_pixels = width * height;
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int block = 256;
		unsigned int grid = nBlock(n_pixels, block);
		colorDepthKernel<<<grid,block>>>(canvas, k, b, inverse,n_pixels);
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail14!\n");
	}

	void visAnisotropyCore(float* fill_color, const float* scale, int n_points, float max_rate) {
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int block = 256;
		unsigned int grid = nBlock(n_points , block);
		visAnisotropyKernel<block> << <grid, block >> > (fill_color, scale, n_points, max_rate);
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail14!\n");
	}	

	void CompositeBasisCore2(
		const float* params,
		const float* transfer,
		// 
		const float* shs_mean,
		const float* opacity_mean,
		const float* rot_mean,
		const float* scale_mean,
		//
		const float* shs_basis,
		const float* opacity_basis,
		const float* rot_basis,
		const float* scale_basis,
		//
		float* shs_tar,
		float* opacity_tar,
		float* rot_tar,
		float* scale_tar,
		int n_points, int n_total_dim, int n_compute_dim,
		bool enable_rot_geom , bool enable_rot_sh 
	) {
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int loop = 4;
		const unsigned int block = 512;
		unsigned int grid;
		if (n_compute_dim <= 256) {
			//CheckCudaError(0);
			
			// NOTICE, here in out is the same may have problem
			if (enable_rot_sh) {
				grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)32);
				CompositeBasisRotSHKernel << <grid, 32 * 3 >> > (
					n_points,transfer, shs_mean, shs_tar
				);
				CheckCudaError(0);
				grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)block * loop);
				CompositeBasisSHKernel_Inplace<loop, 256> << <grid, block >> > (
					n_points * 4 * 4 * 3, 0, n_compute_dim, n_total_dim,
					params, shs_tar, shs_basis
				);
				CheckCudaError(1);
			}
			else {
				grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)block * loop);
				CompositeBasisSHKernel<loop, 256> << <grid, block >> > (
					n_points * 4 * 4 * 3, 0, n_compute_dim, n_total_dim,
					params, shs_mean, shs_basis, shs_tar
				);
			}
			//CheckCudaError(1);
			grid = (unsigned int)nBlock((int64_t)n_points * 3, (int64_t)block * loop);
			CompositeBasisScaleKernel<loop, 256> << <grid, block >> > (
				n_points * 3, 0, n_compute_dim, n_total_dim,
				params, scale_mean, scale_basis, scale_tar
			);
			//CheckCudaError(2);
			grid = (unsigned int)nBlock((int64_t)n_points * 1, (int64_t)block * loop);
			CompositeBasisOpacityKernel<loop, 256> << <grid, block >> > (
				n_points * 1, 0, n_compute_dim, n_total_dim,
				params, opacity_mean, opacity_basis, opacity_tar
				);
			//CheckCudaError(3);
			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
			CompositeBasisRotKernel2<256> << <grid, block >> > (
				n_points, 0, n_compute_dim, n_total_dim, enable_rot_geom,
				params, transfer, rot_mean, rot_basis, rot_tar
			);
			//CheckCudaError(4);
		}
		else {
			throw std::runtime_error("no proper kernel!");
		}
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail12!\n");

		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) throw std::runtime_error("CompositeBasisKernel fail!\n");
	}

	void CompositeBasisCore(
		const float* params,
		//
		const float* shs_mean,
		const float* opacity_mean,
		const float* rot_mean,
		const float* scale_mean,
		//
		const float* shs_basis,
		const float* opacity_basis,
		const float* rot_basis,
		const float* scale_basis,
		//
		float* shs_tar,
		float* opacity_tar,
		float* rot_tar,
		float* scale_tar,
		int n_points, int n_total_dim, int n_compute_dim
	) {
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int loop = 4;
		const unsigned int block = 512;
		unsigned int grid;
		if (n_compute_dim <= 256) {
			//CheckCudaError(0);
			grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)block * loop);
			CompositeBasisSHKernel<loop, 256> << <grid, block >> > (
				n_points * 4 * 4 * 3, 0, n_compute_dim, n_total_dim,
				params, shs_mean, shs_basis, shs_tar
				);
			//CheckCudaError(1);
			grid = (unsigned int)nBlock((int64_t)n_points * 3, (int64_t)block * loop);
			CompositeBasisScaleKernel<loop, 256> << <grid, block >> > (
				n_points * 3, 0, n_compute_dim, n_total_dim,
				params, scale_mean, scale_basis, scale_tar
				);
			//CheckCudaError(2);
			grid = (unsigned int)nBlock((int64_t)n_points * 1, (int64_t)block * loop);
			CompositeBasisOpacityKernel<loop, 256> << <grid, block >> > (
				n_points * 1, 0, n_compute_dim, n_total_dim,
				params, opacity_mean, opacity_basis, opacity_tar
				);
			//CheckCudaError(3);
			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
			CompositeBasisRotKernel<256> << <grid, block >> > (
				n_points, 0, n_compute_dim, n_total_dim,
				params, rot_mean, rot_basis, rot_tar
				);
			//CheckCudaError(4);
		}
		else {
			throw std::runtime_error("no proper kernel!");
		}		
		//cudaStatus = cudaDeviceSynchronize();
		//if (cudaStatus != cudaSuccess) 	throw std::runtime_error("debug fail12!\n");

		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) throw std::runtime_error("CompositeBasisKernel fail!\n");
	}

	//

	void ScaleActivationCore(
		const float* src, float* tar, int n_elements, int tar_offset
	) {
		const unsigned int loop = 4;
		const unsigned int block = 512;
		unsigned int grid;
		grid = (unsigned int)nBlock((int64_t)n_elements * 3, (int64_t)block * loop);
		CopyScaleKernel<loop><<<grid, block >>>(
			n_elements * 3, src, tar + tar_offset * 3, true 
		);
	}
	void OpacityActivationCore(
		const float* src, float* tar, int n_elements, int tar_offset) {
		const unsigned int loop = 4;
		const unsigned int block = 512;
		unsigned int grid;
		grid = (unsigned int)nBlock((int64_t)n_elements * 1, (int64_t)block * loop);
		CopyOpacityKernel<loop><<<grid, block >>>(
			n_elements * 1, src, tar + tar_offset * 1, true
		);
	}
	void RotActivationCore(
		const float* src, float* tar, int n_elements, int tar_offset) {
		const unsigned int block = 256;
		unsigned int grid;
		grid = (unsigned int)nBlock((int64_t)n_elements, (int64_t)block);
		CopyRotKernel<block><<<grid, block >>>(
			n_elements, src, tar + tar_offset * 4, true
		);
	}

	void TransferPosCore(const float* mat4x3, const float* src, float* tar, int n_elements, int tar_offset) {
		const unsigned int block = 256;
		unsigned int grid;
		grid = (unsigned int)nBlock((int64_t)n_elements, (int64_t)block);
		TransferPosKernel<<< grid, block >>>(
			n_elements, mat4x3, src, tar + tar_offset * 3
		);
	}
	void TransferSHCore(const float* mat3x3, const float* src, float* tar, int n_elements, int tar_offset) {
		const unsigned int block = 96;
		unsigned int grid;
		grid = (unsigned int)nBlock((int64_t)n_elements, (int64_t)32);
		TransferSHKernel<<<grid, block >>>(
			n_elements, mat3x3, src, tar + tar_offset * 16 * 3
		);
	}
	void TransferRotActivationCore(const float* qrot4, const float* src, float* tar, int n_elements, int tar_offset) {
		const unsigned int block = 256;
		unsigned int grid;
		grid = (unsigned int)nBlock((int64_t)n_elements, (int64_t)block);
		TransferRotActivationKernel << <grid, block >> > (
			n_elements, qrot4, src, tar + tar_offset * 4
		);
	}

	void CompositeFUPipev1Core(
		const float* params, // 51(expr)
		//
		const float* pos_mean,
		const float* rot_mean,
		const float* scale_mean,
		const float* opacity_mean,
		const float* shs_mean,
		//
		const float* xyz_basis,
		const float* rot_basis,
		const float* scale_basis,
		const float* opacity_basis,
		const float* shs_basis,
		//
		float* pos_tar,
		float* shs_tar,
		float* opacity_tar,
		float* rot_tar,
		float* scale_tar,
		//
		int n_points, int n_total_dim, int n_compute_dim,
		int tar_offset
	) {
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int loop = 4;
		const unsigned int block = 512;
		//const unsigned int block2 = 128;
		unsigned int grid;

		if (n_compute_dim <= 256) {
			
			//// these params(opacity, scale) are not affected by complex mechanism ...
			grid = (unsigned int)nBlock((int64_t)n_points * 3, (int64_t)block * loop);
			CompositeBasisScaleKernel<loop, 256> << <grid, block >> > (
				n_points * 3, 0, n_compute_dim, n_total_dim,
				params, scale_mean, scale_basis, scale_tar + 3 * tar_offset
				);

			grid = (unsigned int)nBlock((int64_t)n_points * 1, (int64_t)block * loop);
			CompositeBasisOpacityKernel<loop, 256> << <grid, block >> > (
				n_points * 1, 0, n_compute_dim, n_total_dim,
				params, opacity_mean, opacity_basis, opacity_tar + 1 * tar_offset
			);

			//// pos

			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
			CompositeBasisPosKernel<256><<<grid,block>>>(
				n_points, 0, n_compute_dim, n_total_dim,
				params, pos_mean, xyz_basis, pos_tar + 3 * tar_offset
			);

			//// rot

			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
			CompositeBasisRotKernel<256> <<<grid, block >>>(
				n_points, 0, n_compute_dim, n_total_dim,
				params, rot_mean, rot_basis, rot_tar + 3 * tar_offset
			);

			//// sh
			if (n_compute_dim == 51 && n_total_dim == 51) {
				grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)64);
				CompositeBasisSHKernelAccN64T64<51> <<<grid, 64 >>>(
					(int64_t)n_points * 4 * 4 * 3, params, shs_mean, shs_basis, shs_tar + tar_offset * 16 * 3 
				);
			}
			else {
				grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)block * loop);
				CompositeBasisSHKernel<loop, 256> <<<grid, block >>> (
					(int64_t)n_points * 4 * 4 * 3, 0, n_compute_dim, n_total_dim,
					params, shs_mean, shs_basis, shs_tar + tar_offset * 16 * 3
				);
			}
		}
		else {
			throw std::runtime_error("no proper kernel!");
		}

	}

	void CompositeNewPipev1Core(
		const float* params, // 100(expr) + 2(eyelid) + 36(pos)
		const float* node_transfer, // 5*4*4
		//
		const float* pos_mean,
		const float* rot_mean,
		const float* scale_mean,
		const float* opacity_mean,
		const float* shs_mean,
		//
		const float* xyz_basis,
		const float* rot_basis,
		const float* scale_basis,
		const float* opacity_basis,
		const float* shs_basis,
		//
		const float* rot_d,
		const float* pos_t,
		const float* W,
		const float* eyelid,
		//
		float* pos_tar,
		float* shs_tar,
		float* opacity_tar,
		float* rot_tar,
		float* scale_tar,
		//
		float ** buffer, size_t * buffer_size,
		bool enable_deform_rot, bool enable_deform_rot_sh,
		bool enable_trans_rot, bool enable_trans_rot_sh,
		int n_points, int n_total_dim, int n_compute_dim,
		int tar_offset, bool force_50
	) {
		cudaError_t cudaStatus = cudaSuccess;
		const unsigned int loop = 4;
		const unsigned int block = 512;
		const unsigned int block2 = 128;
		unsigned int grid;
		if (n_compute_dim + 36 <= 256) {
			
			//CheckCudaError(332);
			//// these params(opacity, scale) are not affected by complex mechanism ...
			grid = (unsigned int)nBlock((int64_t)n_points * 3, (int64_t)block * loop);
			CompositeBasisScaleKernel<loop, 256> << <grid, block >> > (
				n_points * 3, 0, n_compute_dim, n_total_dim,
				params, scale_mean, scale_basis, scale_tar + 3 * tar_offset
			);
			
			grid = (unsigned int)nBlock((int64_t)n_points * 1, (int64_t)block * loop);
			CompositeBasisOpacityKernel<loop, 256> << <grid, block >> > (
				n_points * 1, 0, n_compute_dim, n_total_dim,
				params, opacity_mean, opacity_basis, opacity_tar + 1 * tar_offset
			);
			//CheckCudaError(333);
			//// 1. pos (expr + pose)
			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
			if (force_50) {
				CompositeBasisPosKernel<256> << <grid, block >> > (
					n_points, 0, n_compute_dim, n_total_dim,
					100 + 2, 36,
					params, pos_mean, xyz_basis, pos_t, pos_tar + 3 * tar_offset
				);
			}
			else {
				CompositeBasisPosKernel<256> << <grid, block >> > (
					n_points, 0, 100, 100,
					100 + 2, 36,
					params, pos_mean, xyz_basis, pos_t, pos_tar + 3 * tar_offset
				);
			}

			// allocate buffer
			int n_elem1 = nBlock(12 * n_points,32) * 32; // for byte alignment
			int n_elem2 = nBlock(4 * n_points,32) * 32;
			int n_elem3 = nBlock(4 * n_points,32) * 32;
			float* ptr_root;
			float* ptr_TN;
			float* ptr_DR;
			if (buffer && buffer_size) {
				if (*buffer == nullptr) {
					// allocate first time
					cudaMalloc(buffer, sizeof(float) * (n_elem1 + n_elem2 + n_elem3));
					*buffer_size = n_elem1 + n_elem2 + n_elem3;
				}
				else if ((n_elem1 + n_elem2 + n_elem3) != *buffer_size) {
					// re-allocate
					cudaFree(*buffer);
					cudaMalloc(buffer, sizeof(float) * (n_elem1 + n_elem2 + n_elem3));
					*buffer_size = n_elem1 + n_elem2 + n_elem3;
				}
				ptr_root = *buffer;
			}
			else {
				cudaMalloc(&ptr_root, sizeof(float) * (n_elem1 + n_elem2 + n_elem3));
			}
			ptr_TN = ptr_root + n_elem1;
			ptr_DR = ptr_TN + n_elem2;

			// 2. skinning and compute transfer
			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block2);
			CompositeT<block2><<<grid,block2>>>(
				n_points, W, node_transfer, ptr_root, ptr_TN
			);

			bool compute_deform_rot = enable_deform_rot || enable_deform_rot_sh;
			if (compute_deform_rot && !rot_d) {
				static int warning_flag = 0; // warning only once !
				if (warning_flag == 0) {
					printf("warning! rot_d is not available\n");
					warning_flag = 1;
				}
			}			
			compute_deform_rot = rot_d && compute_deform_rot;

			// 3. [optional]
			if (compute_deform_rot) {
				grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
				CompositeDeformRot<256><<<grid,block>>>(
					//n_points, 0, n_compute_dim, n_total_dim, true
					n_points, 0, 100, 100, true,
					params, rot_d, ptr_DR
				);
			}

			//CheckCudaError(334);

			// 4. apply transfer and add eyelid
			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block2);
			RotPosAndAddEyelidKernel_Inplace<block2><<<grid,block2>>>(
				n_points, 100,
				params, ptr_root, eyelid, pos_tar + 3 * tar_offset
			);

			//////////////////

			// 5. rot: apply deform rot and basis
			grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)block);
			CompositeBasisRotKernel3<256> << <grid, block >> > ( 
				n_points, 0, n_compute_dim, n_total_dim, compute_deform_rot && enable_deform_rot,
				params, rot_mean, rot_basis, ptr_DR, rot_tar + 4 * tar_offset
			);
			//CheckCudaError(335);

			// 6. rot: apply trans rot 
			if (enable_trans_rot) {
				CompositeRotKernel_Inplace<256> << <grid, block >> > (
					n_points, params, ptr_TN, rot_tar + 4 * tar_offset
				);
				//CheckCudaError(336);
			}

			//////////////////

			// 7. sh: apply deform rot 
			if (compute_deform_rot && enable_deform_rot_sh) {
				grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)32);
				TransferSHKernelQuat<< <grid, 96 >> > (
					n_points, ptr_DR, shs_mean, shs_tar + tar_offset * 16 * 3
				);
			}
			else {
				cudaMemcpy(shs_tar + tar_offset * 16 * 3, shs_mean, sizeof(float) * n_points * 16 * 3, cudaMemcpyDeviceToDevice);
			}
			//CheckCudaError(337);
			
			// 8. sh: add basis			
			
			if (n_compute_dim == 50 && n_total_dim == 50) {
				// acceleration but assert n_compute_dim == 50, n_total_dim == 50
				
				//grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)32);
				//CompositeBasisSHKernelAccN32T256_Inplace<50> << <grid, 256 >> > (
				//	(int64_t)n_points * 4 * 4 * 3, params, shs_tar + tar_offset * 16 * 3, shs_basis
				//);	

				//grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)32);
				//CompositeBasisSHKernelAccN32T64_Inplace<50> << <grid, 64 >> > (
				//	(int64_t)n_points * 4 * 4 * 3, params, shs_tar + tar_offset * 16 * 3, shs_basis
				//);

				grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)64);
				CompositeBasisSHKernelAccN64T64_Inplace<50> << <grid, 64 >> > (
					(int64_t)n_points * 4 * 4 * 3, params, shs_tar + tar_offset * 16 * 3, shs_basis
				);

				//grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)128);
				//CompositeBasisSHKernelAccN128T128_Inplace<50> << <grid, 128 >> > (
				//	(int64_t)n_points * 4 * 4 * 3, params, shs_tar + tar_offset * 16 * 3, shs_basis
				//);				

			}
			else {
				grid = (unsigned int)nBlock((int64_t)n_points * 4 * 4 * 3, (int64_t)block * loop);
				CompositeBasisSHKernel_Inplace<loop, 256> << <grid, block >> > (
					(int64_t)n_points * 4 * 4 * 3, 0, n_compute_dim, n_total_dim,
					params, shs_tar + tar_offset * 16 * 3, shs_basis
				);
			}			

			//CheckCudaError(338);
			// 9. sh: apply trans rot 
			if (enable_trans_rot_sh) {
				grid = (unsigned int)nBlock((int64_t)n_points, (int64_t)32);
				TransferSHKernelQuat_Inplace << <grid, 96 >> > (
					n_points, ptr_TN, shs_tar + tar_offset * 16 * 3
				);
			}
			//CheckCudaError(339);

			if (buffer && buffer_size) { /* nothing to do, no need to allocate and release each time */ }
			else { cudaFree(ptr_root); }
		}
		else {
			throw std::runtime_error("no proper kernel!");
		}
	}

}
