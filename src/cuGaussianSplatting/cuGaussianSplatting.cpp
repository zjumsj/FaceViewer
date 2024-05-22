//#include <Windows.h>

#include <math.h>
#include "GL/glew.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdio.h>
//#include <exception>
#include <stdexcept>

#include "cuGaussianSplatting.h"
#include "etc.h"

namespace cuGaussianSplatting {

	struct cudaGraphicsResource* resources[MAX_RESOURCES];
	cudaStream_t streams[MAX_STREAMS];
	int n_streams = 0;

	/////////////////////////
	
	static inline cudaGraphicsRegisterFlags trans(REGISTER_FLAG flag) {
		cudaGraphicsRegisterFlags _flag = cudaGraphicsRegisterFlagsNone;
		switch (flag) {
		case REGISTER_FLAG::NONE:
			_flag = cudaGraphicsRegisterFlagsNone; break;
		case REGISTER_FLAG::READONLY:
			_flag = cudaGraphicsRegisterFlagsReadOnly; break;
		case REGISTER_FLAG::WRITEDISCARD:
			_flag = cudaGraphicsRegisterFlagsWriteDiscard; break;
		case REGISTER_FLAG::SURFACELOADSTORE:
			_flag = cudaGraphicsRegisterFlagsSurfaceLoadStore; break;
		case REGISTER_FLAG::TEXTUREGATHER:
			_flag = cudaGraphicsRegisterFlagsTextureGather; break;
		default:
			break;
		}
		return _flag;
	}

	void RegisterGLRenderBuffersOrTextures(unsigned int object_id, unsigned int gl_enum, REGISTER_FLAG flag, unsigned int unit_idx) {
		cudaError_t cudaStatus = cudaGraphicsGLRegisterImage(&resources[unit_idx], object_id, gl_enum, trans(flag));
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsGLRegisterImage fail!\n");
	}

	void RegisterGLBuffers(unsigned int object_id, REGISTER_FLAG flag, unsigned int unit_idx) {
		cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&resources[unit_idx], object_id, trans(flag));
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsGLRegisterBuffer fail!\n");
	}

	void UnregisterGLResources(unsigned int unit_idx) {
		cudaError_t cudaStatus = cudaGraphicsUnregisterResource(resources[unit_idx]);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsUnregisterResource fail!\n");
	}

	void GLMappedPointer(void** ptr, size_t& bytes, unsigned int unit_idx) {
		cudaError_t cudaStatus = cudaGraphicsMapResources(1, &resources[unit_idx]);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsMapResources fail!\n");
		cudaStatus = cudaGraphicsResourceGetMappedPointer(ptr, &bytes, resources[unit_idx]);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsResourceGetMappedPointer fail!\n");
	}

	void GLUnmapPointer(unsigned int unit_idx) {
		cudaError_t cudaStatus = cudaGraphicsUnmapResources(1, &resources[unit_idx]);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsUnmapResources fail!\n");
	}

	void CheckCudaError(int i) {

		cudaDeviceSynchronize();
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			//char cbuffer[512];
			printf("checkpoint %d %s\n", i, cudaGetErrorString(cudaStatus));
			throw std::runtime_error("CheckCudaError");
		}
		
	}


	///////////////////////////

	int CreateStreams(int N) {
		int n_valid = N > MAX_STREAMS ? MAX_STREAMS : N;
		if (n_valid > n_streams) {
			// add new stream
			for (int i = n_streams; i < n_valid; i++) {
				cudaError_t cudaStatus = cudaStreamCreate(&streams[i]);
				if (cudaStatus != cudaSuccess)
					throw std::runtime_error("cudaStreamCreate fail!\n");
			}
		}
		else if (n_valid < n_streams) {
			// delete streams
			for (int i = n_valid; i < n_streams; i++) {
				cudaError_t cudaStatus = cudaStreamDestroy(streams[i]);
				if (cudaStatus != cudaSuccess)
					throw std::runtime_error("cudaStreamDestroy fail!\n");
			}
		}
		n_streams = n_valid;
		return n_valid;
	}

	int GetStreamNumber() {
		return n_streams;
	}

	void DestroyStreams() {
		for (int i = 0; i < n_streams; i++) {
			cudaError_t cudaStatus = cudaStreamDestroy(streams[i]);
			if (cudaStatus != cudaSuccess)
				throw std::runtime_error("cudaStreamDestroy fail!\n");
		}
		n_streams = 0;
	}

	void SetCudaHeapSize(size_t n) {
		cudaError_t cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, n);
		if (cudaStatus != cudaSuccess) {
			throw std::runtime_error("cudaDeviceSetLimit fail!\n");
		}
	}

	std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S){
		auto lambda = [ptr, &S](size_t N) {
			if (N > S)
			{
				if (*ptr)
					cudaFree(*ptr);
				//CUDA_SAFE_CALL(cudaFree(*ptr));
				cudaMalloc(ptr, 2 * N);
				//CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
				S = 2 * N;
			}
			return reinterpret_cast<char*>(*ptr);
		};
		return lambda;
	}

	void HostToDeviceWrapper(void* tar, const void* src, size_t n_bytes) {
		cudaMemcpy(tar, src, n_bytes, cudaMemcpyHostToDevice);
	}

	void DeviceToHostWrapper(void* tar, const void* src, size_t n_bytes) {
		cudaMemcpy(tar, src, n_bytes, cudaMemcpyDeviceToHost);
	}

	void DeviceToDeviceWrapper(void* tar, const void* src, size_t n_bytes) {
		cudaMemcpy(tar, src, n_bytes, cudaMemcpyDeviceToDevice);
	}

	///////////////////////////
	
	cuGaussianBasic::cuGaussianBasic(){}
	
	cuGaussianBasic::~cuGaussianBasic() {
		//printf("cuGaussianBasic deconstruct!\n");
		Destroy();
	}

	void cuGaussianBasic::initParams(
		int n_points, int color_channel, int l_sh,
		int n_scale_elems, int n_rot_elems
	) {
		this->n_points = n_points;
		this->color_channel = color_channel;
		this->l_sh = l_sh;
		this->n_scale_elems = n_scale_elems;
		this->n_rot_elems = n_rot_elems;
	}

	void cuGaussianBasic::Destroy() {
		//printf("cuGaussianBasic Destroy called!\n");
		if (d_pos) { cudaFree(d_pos); d_pos = nullptr; }
		if (d_rot) { cudaFree(d_rot); d_rot = nullptr; }
		if (d_scale) { cudaFree(d_scale); d_scale = nullptr; }
		if (d_opacity) { cudaFree(d_opacity); d_opacity = nullptr; }
		if (d_shs) { cudaFree(d_shs); d_shs = nullptr; }

		if (d_frameRt_NR) { cudaFree(d_frameRt_NR); d_frameRt_NR = nullptr; }
	}

	void cuGaussianBasic::AllocateData() {
		cudaMalloc(&d_pos, n_points * 3 * sizeof(float));
		cudaMalloc(&d_rot, n_points * 4 * sizeof(float));
		cudaMalloc(&d_scale, n_points * 3 * sizeof(float));
		cudaMalloc(&d_opacity, n_points * 1 * sizeof(float));
		int n_elem_sh = get_elem_sh();
		cudaMalloc(&d_shs, n_points * n_elem_sh * sizeof(float));
	}

	void cuGaussianBasic::CopyData(
		float * pos, float * rot, float * scale,
		float * opacity, float * shs
	) {
		if (pos) {	cudaMemcpy(d_pos, pos, n_points * 3 * sizeof(float), cudaMemcpyHostToDevice); }
		if (rot) {	cudaMemcpy(d_rot, rot, n_points * 4 * sizeof(float), cudaMemcpyHostToDevice); }
		if (scale) { cudaMemcpy(d_scale, scale, n_points * 3 * sizeof(float), cudaMemcpyHostToDevice); }
		if (opacity) {	cudaMemcpy(d_opacity, opacity, n_points * 1 * sizeof(float), cudaMemcpyHostToDevice); }
		int n_elem_sh = get_elem_sh();
		if (shs) {	cudaMemcpy(d_shs, shs, n_points * n_elem_sh * sizeof(float), cudaMemcpyHostToDevice); }
	}

	void cuGaussianBasic::allocateTransferData() {
		cudaMalloc((void**)&d_frameRt_NR, sizeof(float) * (16 + 16 + 16));
	}

	void matrix_to_quaternion(float* outq, const float* mat) {
		float q_abs[4];
		q_abs[0] = 1 + mat[0] + mat[4] + mat[8];
		q_abs[1] = 1 + mat[0] - mat[4] - mat[8];
		q_abs[2] = 1 - mat[0] + mat[4] - mat[8];
		q_abs[3] = 1 - mat[0] - mat[4] + mat[8];
		for (int i = 0; i < 4; i++) {
			if (q_abs[i] > 0) q_abs[i] = sqrt(q_abs[i]);
			else q_abs[i] = 0;
		}
		float div;
		int argmax_id = 0;
		float max_v = q_abs[0];
		if (q_abs[1] > max_v) { max_v = q_abs[1]; argmax_id = 1; }
		if (q_abs[2] > max_v) { max_v = q_abs[2]; argmax_id = 2; }
		if (q_abs[3] > max_v) { max_v = q_abs[3]; argmax_id = 3; }
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
		}
		else {
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

	void cuGaussianBasic::copyTransferData(const float * frameRt, const float * frameNR) {
		float buffer[16 + 16 + 16];
		for (int i = 0; i < 3 * 4; i++)
			buffer[i] = frameRt[i];// 3x4
		for (int i = 0; i < 3 * 3; i++)
			buffer[16 + i] = frameNR[i]; // 3x3

		matrix_to_quaternion(&buffer[32],frameNR); // 3x3->4
		cudaMemcpy(d_frameRt_NR, buffer, sizeof(float) * 16 * 3, cudaMemcpyHostToDevice);
	}

	//

	cuGaussianFace::cuGaussianFace() :cuGaussianBasic() {}

	cuGaussianFace::~cuGaussianFace() {
		//printf("cuGaussianFace deconstruct!\n");
		Destroy();
	}

	void cuGaussianFace::Destroy() {
		//printf("cuGaussianFace Destroy called!\n");
		if (d_xyz_t) { cudaFree(d_xyz_t); d_xyz_t = nullptr; }
		if (d_rot_t) { cudaFree(d_rot_t); d_rot_t = nullptr; }
		if (d_scale_t) { cudaFree(d_scale_t); d_scale_t = nullptr; }
		if (d_opacity_t) { cudaFree(d_opacity_t); d_opacity_t = nullptr; }
		if (d_shs_t) { cudaFree(d_shs_t); d_shs_t = nullptr; }
		
		if (d_rot_d) { cudaFree(d_rot_d); d_rot_d = nullptr; }
		if (d_pos_t) { cudaFree(d_pos_t); d_pos_t = nullptr; }
		if (d_W) { cudaFree(d_W); d_W = nullptr; }
		if (d_eyelid) { cudaFree(d_eyelid); d_eyelid = nullptr; }
		// extra
		if (d_node_transfer) { cudaFree(d_node_transfer); d_node_transfer = nullptr; }
		if (d_buffer) { cudaFree(d_buffer); d_buffer = nullptr; d_buffer_size = 0; }
	}

	void cuGaussianFace::AllocateData(bool alloc_rot_d) {
		cuGaussianBasic::AllocateData();
		if (force_50) {
			cudaMalloc(&d_xyz_t, n_points * 3 * n_basis * sizeof(float));
		}
		else {
			cudaMalloc(&d_xyz_t, n_points * 3 * 100 * sizeof(float));
		}
		cudaMalloc(&d_rot_t, n_points * 4 * n_basis * sizeof(float));
		cudaMalloc(&d_scale_t, n_points * 3 * n_basis * sizeof(float));
		cudaMalloc(&d_opacity_t, n_points * 1 * n_basis * sizeof(float));
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		cudaMalloc(&d_shs_t, n_points * n_elem_sh * n_basis * sizeof(float));

		if (alloc_rot_d) {
			//cudaMalloc(&d_rot_d, n_points * 4 * n_basis * sizeof(float));
			cudaMalloc(&d_rot_d, n_points * 4 * 100 * sizeof(float));
		}
		cudaMalloc(&d_pos_t, 36 * n_points * 3 * sizeof(float));
		cudaMalloc(&d_W, n_points * 5 * sizeof(float));
		cudaMalloc(&d_eyelid, n_points * 3 * 2 * sizeof(float));
	}

	void cuGaussianFace::AllocateDataSimple() {
		cuGaussianBasic::AllocateData();
		cudaMalloc(&d_xyz_t, n_points * 3 * n_basis * sizeof(float));
		cudaMalloc(&d_rot_t, n_points * 4 * n_basis * sizeof(float));
		cudaMalloc(&d_scale_t, n_points * 3 * n_basis * sizeof(float));
		cudaMalloc(&d_opacity_t, n_points * 1 * n_basis * sizeof(float));
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		cudaMalloc(&d_shs_t, n_points * n_elem_sh * n_basis * sizeof(float));
	}

	void cuGaussianFace::CopyDataEx(
		float * xyz_t, float * rot_t, float * scale_t, float * opacity_t, float * shs_t,
		float * rot_d, float * pos_t, float * W, float * eyelid
	) {
		if (force_50) {
			if (xyz_t) { cudaMemcpy(d_xyz_t, xyz_t, n_points * 3 * n_basis * sizeof(float), cudaMemcpyHostToDevice); }
		}
		else {
			if (xyz_t) { cudaMemcpy(d_xyz_t, xyz_t, n_points * 3 * 100 * sizeof(float), cudaMemcpyHostToDevice); }
		}
		if (rot_t) { cudaMemcpy(d_rot_t, rot_t, n_points * 4 * n_basis * sizeof(float), cudaMemcpyHostToDevice); }
		if (scale_t) { cudaMemcpy(d_scale_t, scale_t, n_points * 3 * n_basis * sizeof(float), cudaMemcpyHostToDevice); }
		if (opacity_t) { cudaMemcpy(d_opacity_t, opacity_t, n_points * 1 * n_basis * sizeof(float), cudaMemcpyHostToDevice); }
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		if (shs_t) { cudaMemcpy(d_shs_t, shs_t, n_points * n_elem_sh * n_basis * sizeof(float), cudaMemcpyHostToDevice); }

		//if (rot_d) { cudaMemcpy(d_rot_d, rot_d, n_points * 4 * n_basis * sizeof(float), cudaMemcpyHostToDevice); }
		if (rot_d) { cudaMemcpy(d_rot_d, rot_d, n_points * 4 * 100 * sizeof(float), cudaMemcpyHostToDevice); }
		if (pos_t) { cudaMemcpy(d_pos_t, pos_t, 36 * n_points * 3 * sizeof(float), cudaMemcpyHostToDevice); }
		if (W) { cudaMemcpy(d_W, W, n_points * 5 * sizeof(float), cudaMemcpyHostToDevice); }
		if (eyelid) { cudaMemcpy(d_eyelid, eyelid, n_points * 3 * 2 * sizeof(float), cudaMemcpyHostToDevice); }
	}

	void cuGaussianFace::allocateNodeTransfer() {
		const int n_node = 5;
		cudaMalloc((void**)&d_node_transfer, sizeof(float) * (n_node * 4 * 4));	
	}

	void cuGaussianFace::copyNodeTransfer(const float* node_transfer) {
		const int n_node = 5;
		cudaMemcpy(d_node_transfer, node_transfer, sizeof(float) * (n_node * 4 * 4), cudaMemcpyHostToDevice);
	}

	///////////////////////////

	GPoints::GPoints() {}
	void GPoints::initParams(
		int n_points, int color_channel, int l_sh,
		int n_scale_elems, int n_rot_elems
	) {
		this->n_points = n_points;
		this->color_channel = color_channel;
		this->l_sh = l_sh;
		this->n_scale_elems = n_scale_elems;
		this->n_rot_elems = n_rot_elems;
	}
	void GPoints::copyOnlyPos(float* pos) {
		cudaMemcpy(pos_cuda, pos, sizeof(float) * color_channel * n_points, cudaMemcpyHostToDevice);
	}

	void GPoints::copyTransferData(float* transfer) {
		cudaMemcpy(transfer_cuda, transfer, sizeof(float) * 9 * n_points, cudaMemcpyHostToDevice);
	}

	void GPoints::fillColor(float* color3) {
		if (color_cuda) {
			fillColorCore(color_cuda, n_points, color3);
		}
		else {
			printf("No valid color buffer!");
		}
	}

	void GPoints::fillColorWithAnisotropy(float max_rate) {
		if (color_cuda && scale_cuda) {
			visAnisotropyCore(color_cuda, scale_cuda, n_points, max_rate);
		}
		else {
			printf("No valid scale or color buffer!");
		}
	}

	void GPoints::fillColorWithDepth(float* modelview) {
		if (color_cuda) {
			//if (buffer_cuda == 0) {
			//	cudaMalloc((void**)&buffer_cuda, 128); // 128 bytes
			//}
			fillColorWithDepthCore(color_cuda, pos_cuda, n_points, modelview);
		}
		else {
			printf("No valid color buffer!");
		}
	}

	void GPoints::ColorDepth(float* color, int width, int height, float k, float b, bool inverse) {
		ColorDepthCore(color, width, height, k, b, inverse);
	}

	void GPoints::debug_dump(float** x, int& n, int type_) {
		if (type_ == 0) { // pos
			n = 3 * n_points;
			*x = new float[n];
			cudaMemcpy((void*)*x, pos_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		}
		else if (type_ == 1) {//shs
			int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
			n = n_elem_sh * n_points;
			*x = new float[n];
			cudaMemcpy((void*)*x, shs_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		}
		else if (type_ == 2) {//opacity
			n = n_points;
			*x = new float[n];
			cudaMemcpy((void*)*x, opacity_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		}
		else if (type_ == 3) {//rot
			n = n_points * n_rot_elems;
			*x = new float[n];
			cudaMemcpy((void*)*x, rot_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		}
		else if (type_ == 4) {//scale
			n = n_points * n_scale_elems;
			*x = new float[n];
			cudaMemcpy((void*)*x, scale_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		}
	}
	void GPoints::allocateData(
		bool pos, bool color, bool shs,
		bool opacity, bool normal, bool rot, bool scale
	) {
		if (pos) {
			cudaMalloc((void**)&pos_cuda, sizeof(float) * 3 * n_points);
		}
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		if (color) {
			cudaMalloc((void**)&color_cuda, sizeof(float) * color_channel * n_points);
		}
		if (shs) {
			cudaMalloc((void**)&shs_cuda, sizeof(float) * n_elem_sh * n_points);
		}
		if (opacity) {
			cudaMalloc((void**)&opacity_cuda, sizeof(float) * 1 * n_points);
		}
		if (normal) {
			cudaMalloc((void**)&normal_cuda, sizeof(float) * 3 * n_points);
		}
		if (rot) {
			cudaMalloc((void**)&rot_cuda, sizeof(float) * n_rot_elems * n_points);
		}
		if (scale) {
			cudaMalloc((void**)&scale_cuda, sizeof(float) * n_scale_elems * n_points);
		}
	}
	void GPoints::copyData(
		float* pos, float* color, float* shs,
		float* opacity, float* normal,
		float* rot, float* scale
	) {
		if (pos) {
			cudaMemcpy(pos_cuda, pos, sizeof(float) * color_channel * n_points, cudaMemcpyHostToDevice);
		}
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		if (color) {
			cudaMemcpy(color_cuda, color, sizeof(float) * color_channel * n_points, cudaMemcpyHostToDevice);
		}
		if (shs) {
			cudaMemcpy(shs_cuda, shs, sizeof(float) * n_elem_sh * n_points, cudaMemcpyHostToDevice);
		}
		if (opacity) {
			cudaMemcpy(opacity_cuda, opacity, sizeof(float) * 1 * n_points, cudaMemcpyHostToDevice);
		}
		if (normal) {
			cudaMemcpy(normal_cuda, normal, sizeof(float) * 3 * n_points, cudaMemcpyHostToDevice);
		}
		if (rot) {
			cudaMemcpy(rot_cuda, rot, sizeof(float) * n_rot_elems * n_points, cudaMemcpyHostToDevice);
		}
		if (scale) {
			cudaMemcpy(scale_cuda, scale, sizeof(float) * n_scale_elems * n_points, cudaMemcpyHostToDevice);
		}
	}
	void GPoints::allocateCopyData(
		float* pos, float* color, float* shs,
		float* opacity, float* normal,
		float* rot, float* scale
	) {
		if (pos) {
			cudaMalloc((void**)&pos_cuda, sizeof(float) * color_channel * n_points);
			cudaMemcpy(pos_cuda, pos, sizeof(float) * color_channel * n_points, cudaMemcpyHostToDevice);
		}
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		if (color) {
			cudaMalloc((void**)&color_cuda, sizeof(float) * color_channel * n_points);
			cudaMemcpy(color_cuda, color, sizeof(float) * color_channel * n_points, cudaMemcpyHostToDevice);
		}
		if (shs) {
			cudaMalloc((void**)&shs_cuda, sizeof(float) * n_elem_sh * n_points);
			cudaMemcpy(shs_cuda, shs, sizeof(float) * n_elem_sh * n_points, cudaMemcpyHostToDevice);
		}
		if (opacity) {
			cudaMalloc((void**)&opacity_cuda, sizeof(float) * 1 * n_points);
			cudaMemcpy(opacity_cuda, opacity, sizeof(float) * 1 * n_points, cudaMemcpyHostToDevice);
		}
		if (normal) {
			cudaMalloc((void**)&normal_cuda, sizeof(float) * 3 * n_points);
			cudaMemcpy(normal_cuda, normal, sizeof(float) * 3 * n_points, cudaMemcpyHostToDevice);
		}
		if (rot) {
			cudaMalloc((void**)&rot_cuda, sizeof(float) * n_rot_elems * n_points);
			cudaMemcpy(rot_cuda, rot, sizeof(float) * n_rot_elems * n_points, cudaMemcpyHostToDevice);
		}
		if (scale) {
			cudaMalloc((void**)&scale_cuda, sizeof(float) * n_scale_elems * n_points);
			cudaMemcpy(scale_cuda, scale, sizeof(float) * n_scale_elems * n_points, cudaMemcpyHostToDevice);
		}
	}

	void GPoints::allocateRectData() {
		cudaMalloc((void**)&rect_cuda, sizeof(float) * 2 * n_points);
	}

	void GPoints::allocateTransferData() {
		cudaMalloc((void**)&transfer_cuda, sizeof(float) * 9 * n_points);
	}

	void GPoints::allocateBkgData(bool white, float* replace) {

		float bkg_color[3];
		if (replace) {
			bkg_color[0] = replace[0];
			bkg_color[1] = replace[1];
			bkg_color[2] = replace[2];
		}
		else {
			if (white) {
				bkg_color[0] = bkg_color[1] = bkg_color[2] = 1.f;
			}
			else {
				bkg_color[0] = bkg_color[1] = bkg_color[2] = 0.f;
			}
		}
		if (bkg_cuda == nullptr) {
			cudaMalloc((void**)&bkg_cuda, sizeof(float) * 3);
		}
		cudaMemcpy(bkg_cuda, bkg_color, 3 * sizeof(float), cudaMemcpyHostToDevice);
	}

	void GPoints::Destroy() {
		if (pos_cuda) { cudaFree(pos_cuda); pos_cuda = nullptr; }
		if (color_cuda) { cudaFree(color_cuda); color_cuda = nullptr; }
		if (shs_cuda) { cudaFree(shs_cuda); shs_cuda = nullptr; }
		if (opacity_cuda) { cudaFree(opacity_cuda); opacity_cuda = nullptr; }
		if (normal_cuda) { cudaFree(normal_cuda); normal_cuda = nullptr; }
		if (rot_cuda) { cudaFree(rot_cuda); rot_cuda = nullptr; }
		if (scale_cuda) { cudaFree(scale_cuda); scale_cuda = nullptr; }
		//
		if (transfer_cuda) { cudaFree(transfer_cuda); transfer_cuda = nullptr; }
		//
		if (rect_cuda) { cudaFree(rect_cuda); rect_cuda = nullptr; }
		if (bkg_cuda) { cudaFree(bkg_cuda); bkg_cuda = nullptr; }
		//if (buffer_cuda) { cudaFree(buffer_cuda); buffer_cuda = nullptr; }
	}

	GPoints::~GPoints() {
		Destroy();
	}
	///////////////////////////
	
	GPoints_basis::~GPoints_basis() {
		Destroy();
	}

	void GPoints_basis::debug_dump(float** x, int& n, int type_) {
		//if (type_ == 0) { // pos
		//	n = 3 * n_points * n_basis;
		//	*x = new float[n];
		//	cudaMemcpy((void*)*x, pos_t_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		//}
		if (type_ == 1) { // shs
			int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
			n = n_elem_sh * n_points * n_basis;
			*x = new float[n];
			cudaMemcpy((void*)*x, shs_t_cuda, sizeof(float) * n, cudaMemcpyDeviceToHost);
		}
	}

	void GPoints_basis::initParams(
		int n_points, int n_basis, int color_channel, int l_sh,
		int n_scale_elems, int n_rot_elems
	) {
		this->n_points = n_points;
		this->n_basis = n_basis;
		this->color_channel = color_channel;
		this->l_sh = l_sh;
		this->n_scale_elems = n_scale_elems;
		this->n_rot_elems = n_rot_elems;
	}

	void GPoints_basis::copyData(
		float* pos_t, float* color_t,
		float* shs_t, float* opacity_t,
		float* rot_t, float* scale_t
	) {
		if (pos_t) {
			cudaMalloc((void**)&pos_t_cuda, sizeof(float) * 3 * n_points * n_basis);
			cudaMemcpy(pos_t_cuda, pos_t, sizeof(float) * 3 * n_points * n_basis, cudaMemcpyHostToDevice);
		}
		int n_elem_sh = (l_sh + 1) * (l_sh + 1) * color_channel;
		if (color_t) {
			cudaMalloc((void**)&color_t_cuda, sizeof(float) * color_channel * n_points * n_basis);
			cudaMemcpy(color_t_cuda, color_t, sizeof(float) * color_channel * n_points * n_basis, cudaMemcpyHostToDevice);
		}
		if (shs_t) {
			cudaMalloc((void**)&shs_t_cuda, sizeof(float) * n_elem_sh * n_points * n_basis);
			cudaMemcpy(shs_t_cuda, shs_t, sizeof(float) * n_elem_sh * n_points * n_basis, cudaMemcpyHostToDevice);
		}
		if (opacity_t) {
			cudaMalloc((void**)&opacity_t_cuda, sizeof(float) * 1 * n_points * n_basis);
			cudaMemcpy(opacity_t_cuda, opacity_t, sizeof(float) * 1 * n_points * n_basis, cudaMemcpyHostToDevice);
		}
		//if (normal) {
		//	cudaMalloc((void**)&normal_cuda, sizeof(float) * 3 * n_points * n_basis);
		//	cudaMemcpy(normal_cuda, normal, sizeof(float) * 3 * n_points * n_basis, cudaMemcpyHostToDevice);
		//}
		if (rot_t) {
			cudaMalloc((void**)&rot_t_cuda, sizeof(float) * n_rot_elems * n_points * n_basis);
			cudaMemcpy(rot_t_cuda, rot_t, sizeof(float) * n_rot_elems * n_points * n_basis, cudaMemcpyHostToDevice);
		}
		if (scale_t) {
			cudaMalloc((void**)&scale_t_cuda, sizeof(float) * n_scale_elems * n_points * n_basis);
			cudaMemcpy(scale_t_cuda, scale_t, sizeof(float) * n_scale_elems * n_points * n_basis, cudaMemcpyHostToDevice);
		}
	}

	void GPoints_basis::Destroy() {
		if (pos_t_cuda) { cudaFree(pos_t_cuda); pos_t_cuda = nullptr; }
		if (color_t_cuda) { cudaFree(color_t_cuda); color_t_cuda = nullptr; }
		if (shs_t_cuda) { cudaFree(shs_t_cuda); shs_t_cuda = nullptr; }
		if (opacity_t_cuda) { cudaFree(opacity_t_cuda); opacity_t_cuda = nullptr; }
		if (rot_t_cuda) { cudaFree(rot_t_cuda); rot_t_cuda = nullptr; }
	}
	
	///////////////////////////
	// old

	void CompositeBasis2(
		const float* params,
		const GPoints* mean,
		const GPoints_basis* basis,
		GPoints* result,
		int n_compute_dim,
		bool enable_rot_geom, bool enable_rot_sh
	) {
		CompositeBasisCore2(
			params, mean->transfer_cuda,
			mean->shs_cuda, mean->opacity_cuda, mean->rot_cuda, mean->scale_cuda,
			basis->shs_t_cuda, basis->opacity_t_cuda, basis->rot_t_cuda, basis->scale_t_cuda,
			result->shs_cuda, result->opacity_cuda, result->rot_cuda, result->scale_cuda,
			basis->n_points, basis->n_basis, n_compute_dim,
			enable_rot_geom, enable_rot_sh
		);
	}

	void CompositeBasis(
		const float * params,
		const GPoints * mean,
		const GPoints_basis * basis,
		GPoints * result, 
		int n_compute_dim
	) {
		CompositeBasisCore(
			params, 
			mean->shs_cuda, mean->opacity_cuda, mean->rot_cuda, mean->scale_cuda,
			basis->shs_t_cuda, basis->opacity_t_cuda, basis->rot_t_cuda, basis->scale_t_cuda,
			result->shs_cuda, result->opacity_cuda, result->rot_cuda, result->scale_cuda,
			basis->n_points, basis->n_basis, n_compute_dim
		);
	}
	
	////////
	// new

	void ScaleActivation(
		const float* src, float* tar, int n_elements, int tar_offset
	) {	ScaleActivationCore(src, tar, n_elements, tar_offset); }

	void OpacityActivation(
		const float* src, float* tar, int n_elements, int tar_offset
	) {	OpacityActivationCore(src, tar, n_elements, tar_offset);}

	void RotActivation(
		const float* src, float* tar, int n_elements, int tar_offset
	) {	RotActivationCore(src, tar, n_elements, tar_offset);}

	void SimpleTransferBasic(
		const cuGaussianBasic* basic,
		GPoints* result,
		bool enable_rot, bool enable_rot_sh,
		int tar_offset
	){
		int n_points = basic->n_points;
		// transfer pos
		TransferPosCore(
			&basic->d_frameRt_NR[0], basic->d_pos, result->pos_cuda, n_points, tar_offset
		);
		// transfer rot
		if (enable_rot) {
			TransferRotActivationCore(
				&basic->d_frameRt_NR[32], basic->d_rot, result->rot_cuda, n_points, tar_offset
			);
		}
		else {
			RotActivationCore(
				basic->d_rot, result->rot_cuda, n_points, tar_offset
			);
		}
		// transfer rot_sh
		if (enable_rot_sh) {
			TransferSHCore(
				&basic->d_frameRt_NR[16], basic->d_shs, result->shs_cuda, n_points, tar_offset
			);
		}
		else {
			cudaMemcpy(
				result->shs_cuda + 16 * 3 * tar_offset,
				basic->d_shs,
				n_points, cudaMemcpyDeviceToDevice
			);
		}
	}

	void CompositeNewPipev1(
		const float* params,
		const float* node_transfer,
		const cuGaussianFace* face,
		GPoints* result,
		int n_compute_dim,
		bool enable_deform_rot, bool enable_deform_rot_sh,
		bool enable_trans_rot, bool enable_trans_rot_sh,
		int tar_offset, bool force_50
	) {
		CompositeNewPipev1Core(
			params, node_transfer,
			face->d_pos, face->d_rot, face->d_scale, face->d_opacity, face->d_shs,
			face->d_xyz_t, face->d_rot_t, face->d_scale_t, face->d_opacity_t, face->d_shs_t,
			face->d_rot_d, face->d_pos_t, face->d_W, face->d_eyelid,
			result->pos_cuda, result->shs_cuda, result->opacity_cuda, result->rot_cuda, result->scale_cuda,
			//nullptr, nullptr, // old behavior
			&(face->d_buffer), &(face->d_buffer_size), // new behavior
			enable_deform_rot, enable_deform_rot_sh,
			enable_trans_rot, enable_trans_rot_sh,
			face->n_points, face->n_basis, n_compute_dim, tar_offset, force_50
		);
	}

	void CompositeFUPipev1(
		const float* params,
		const cuGaussianFace* face,
		GPoints* result,
		int n_compute_dim,
		int tar_offset
	) {
		CompositeFUPipev1Core(
			params,
			face->d_pos, face->d_rot, face->d_scale, face->d_opacity, face->d_shs,
			face->d_xyz_t, face->d_rot_t, face->d_scale_t, face->d_opacity_t, face->d_shs_t,
			result->pos_cuda, result->shs_cuda, result->opacity_cuda, result->rot_cuda, result->scale_cuda,
			face->n_points, face->n_basis, n_compute_dim, tar_offset
		);
	}

	///////////////////////////
	 
	bool cuBuffer::Initialize(int n_byte) {
		Destroy();
		this->n_byte = n_byte;

		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&params, n_byte);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
		return true;
	}

	void cuBuffer::Copy(void * src_data, int n_byte) {
		cudaMemcpy(params, src_data, n_byte, cudaMemcpyHostToDevice);
	}

	void cuBuffer::CopyBack(void* tar_data, int n_byte) {
		cudaMemcpy(tar_data, params, n_byte, cudaMemcpyDeviceToHost);
	}

	void cuBuffer::Destroy() {
		cudaFree(params);
	}

	cuBuffer::~cuBuffer() {
		Destroy();
	}


	///////////////////////////////

	bool Transforms::Initialize() {
		Destroy();

		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&matrices, 128 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
		return true;
	}

	void Transforms::Destroy() {
		if (matrices == nullptr) {
			cudaFree(matrices);
			matrices = nullptr;
		}
	}

	bool Transforms::CopyData() {
		if (matrices == nullptr) return false;
		float* tmp = new float[128];
		for (int i = 0; i < 16; i++)
			tmp[i] = mvpMatrix[i];
		for (int i = 0; i < 16; i++)
			tmp[i + 16] = mvMatrix[i];
		for (int i = 0; i < 16; i++)
			tmp[i + 32] = c2wMatrix[i];
		for (int i = 0; i < 16; i++)
			tmp[i + 48] = projMatrix[i];

		float* int_as_float = (float*)vpParams;
		for (int i = 0; i < 4; i++)
			tmp[i + 64] = int_as_float[i];

		cudaError_t cudaStatus;
		//cudaStatus = cudaMalloc((void **)&matrices, 128 * sizeof(float));
		//if (cudaStatus != cudaSuccess) {
		//	fprintf(stderr, "cudaMalloc failed!");
		//	return false;
		//}
		cudaStatus = cudaMemcpy(matrices, tmp, 128 * sizeof(float), cudaMemcpyHostToDevice);
		delete[] tmp;
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return false;
		}
		return true;
	}

	Transforms::~Transforms() {
		Destroy();
	}

	///////////////////////////

	bool Canvas::Create(int width, int height, int channel) {
		Destroy();

		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc((void**)&canvas, channel * width * height * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
		this->width = width;
		this->height = height;
		this->channel = channel;
		return true;
	}

	void Canvas::Destroy() {
		if (canvas) {
			cudaFree(canvas);
			canvas = nullptr;
		}
	}

	Canvas::~Canvas() {
		Destroy();
	}

	///////////////////////////

	cuBlendTex::cuBlendTex()
		:TexID(0), width(0), height(0), c(0), k(0),
		mean(0), basis(0), has_mean(false)
	{}
	
	cuBlendTex::~cuBlendTex() { Destroy(); }

	bool cuBlendTex::Initialize(int w, int h, int channel, int n_basis) {
		Destroy();
		width = w;
		height = h;
		c = channel;
		k = n_basis;
		// allocate CUDA buffer
		cudaError_t cudaStatus;
		cudaStatus = cudaMalloc(&mean, width * height * c * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
		cudaStatus = cudaMalloc(&basis, width * height * k * c * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
		cudaStatus = cudaMalloc(&params, k * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return false;
		}
		// allocate Texture
		glGenTextures(1, &TexID);
		glBindTexture(GL_TEXTURE_2D, TexID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0);
		glGenerateMipmap(GL_TEXTURE_2D); // TODO , call me after each blend
		
		// Register for cuda use
		cudaStatus = cudaGraphicsGLRegisterImage(&resources[register_slot], TexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		if (cudaStatus != cudaSuccess)
			throw std::runtime_error("cudaGraphicsGLRegisterImage fail!\n");
		
		glBindTexture(GL_TEXTURE_2D, 0);
		return true;
	}

	bool cuBlendTex::Blend(const float* params_, int n) {
		cudaError_t cudaStatus;
		if (n > k) n = k;
		cudaStatus = cudaMemcpy(params, params_, n * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return false;
		}
		//assert(c == 3);
		BlendTex(
			height, width, 0, n, k,
			basis, has_mean ? mean : nullptr, params, register_slot,
			1/255.f,true
		);
		glBindTexture(GL_TEXTURE_2D, TexID);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
		return true;
	}

	bool cuBlendTex::CopyData(const float* basis_, const float* mean_) {
		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpy(basis, basis_, size_t(width) * height * c * k *  sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return false;
		}
		if (mean_) {
			cudaStatus = cudaMemcpy(mean, mean_, size_t(width) * height * c * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!\n");
				return false;
			}
			has_mean = true;
		}
		return true;
	}

	bool cuBlendTex::CopyDataD(const double* basis_, const double* mean_,bool flip) {
		
		float* f_basis_ = new float[size_t(width) * height * c * k];
		if (flip) {
			size_t n_elem = size_t(width) * c * k;
			for (int ih = 0; ih < height ; ih++) {
				for (size_t iw = 0; iw < n_elem; iw++) {
					f_basis_[(height - 1 - ih) * n_elem + iw] = basis_[ih * n_elem + iw];
				}
			}			
		}
		else {
			for (size_t i = 0; i < size_t(width) * height * c * k; i++) {
				f_basis_[i] = basis_[i];
			}
		}
		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpy(basis, f_basis_, size_t(width) * height * c * k * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return false;
		}
		delete[] f_basis_;
		////////
		if (mean_) {
			float* f_mean_ = new float[size_t(width) * height * c];
			if (flip) {
				size_t n_elem = size_t(width) * c;
				for (int ih = 0; ih < height; ih++) {
					for (size_t iw = 0; iw < n_elem; iw++) {
						f_mean_[(height - 1 - ih) * n_elem + iw] = mean_[ih * n_elem + iw];
					}
				}
			}
			else {
				for (size_t i = 0; i < size_t(width) * height * c; i++) {
					f_mean_[i] = mean_[i];
				}
			}
			cudaStatus = cudaMemcpy(mean, f_mean_, size_t(width) * height * c * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!\n");
				return false;
			}
			delete[] f_mean_;
			has_mean = true;
		}		
		return true;
	}

	void cuBlendTex::Destroy() {
		
		if (TexID) {
			// Unregister and delete texture
			cudaGraphicsUnregisterResource(resources[register_slot]);
			glDeleteTextures(1, &TexID); TexID = 0;
		}
		////
		if (mean) { cudaFree(mean); mean = nullptr; }
		if (basis) { cudaFree(basis); basis = nullptr; }
		if (params) { cudaFree(params); params = nullptr; }
		has_mean = false;
	}

}