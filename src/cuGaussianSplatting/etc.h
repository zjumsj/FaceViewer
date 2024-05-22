#pragma once

namespace cuGaussianSplatting {
		
	void BlendTex(
		int height, int width, int p_offset, int p_n, int p_total,
		void* basis, void* means, void* params, int register_idx, 
		float scale, bool bgr
	);

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
		bool enable_rot_geom = true, bool enable_rot_sh = true
	);

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
	);

	void visAnisotropyCore(float* fill_color, const float* scale, int n_points, float max_rate);
	void fillColorWithDepthCore(float* gpu_buffer, const float* pos, int n_points, const float* modelview);
	void ColorDepthCore(float* canvas, int width, int height, float k, float b, bool inverse);
	void fillColorCore(float* gpu_buffer, int n_points, const float* cpu_color3);

	void ScaleActivationCore(const float* src, float* tar, int n_elements, int tar_offset);
	void OpacityActivationCore(const float* src, float* tar, int n_elements, int tar_offset);
	void RotActivationCore(const float* src, float* tar, int n_elements, int tar_offset);
	
	void TransferPosCore(const float* mat4x3, const float* src, float * tar, int n_elements, int tar_offset);
	void TransferSHCore(const float* mat3x3, const float* src, float* tar, int n_elements, int tar_fofset);
	void TransferRotActivationCore(const float* qrot4, const float* src, float* tar, int n_elements, int tar_offset);
	
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
	);

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
	);
}