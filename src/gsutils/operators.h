#pragma once

#include <string>
#include <vector>

#include "../cuGaussianSplatting/cuGaussianSplatting.h"
#include "containers.h"

class SMesh;

namespace gaussian_splatting {
	
	void SetGLTransforms(cuGaussianSplatting::Transforms& trans, bool trans_to_gpu = true);

	// Some dirty coordinate transforms only for gaussian splatting...
	void SetGLTransforms2(cuGaussianSplatting::Transforms& trans, bool trans_to_gpu = true);

	////////////////////

	// matrix in col major
	void perspective_2_orthogonal(
		const float* align_point,
		const float* w2c,
		const float* proj, float* orthogonal,
		float halfD
	);

	Eigen::Vector3f closesPointOnTriangle(
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		const Eigen::Vector3f& sourcePosition,
		float * oS, float* oT
	);


	void GeneratePointPos(
		const SMesh * mesh,
		const gaussian_splatting::gaussian_points_coord& gcoord,
		gaussian_splatting::gaussian_points& gpc
	);

	// 1. compute local frame
	// 2. compute faceR, facet
	// 3. compute faceNR
	void GeneratePointPos(
		const SMesh* mesh,
		const gaussian_splatting::gaussian_points_coord& gcoord,
		local_transfer* loc_transfer,
		gaussian_splatting::gaussian_points& gpc,
		float k
	);

	void remove_expr_avg(
		std::vector<Eigen::Vector3f>& out_vertex,
		const FLAME* flame,
		const float* expr_params, int n_expr_params,
		float hack_scale = 1.f
	);

	void rot_match_back_head(
		const std::vector<Eigen::Vector3f>& A,
		const std::vector<Eigen::Vector3f>& B,
		Eigen::Matrix3f& R,
		Eigen::Vector3f& t
	);

	void get_index(std::vector<int>& index, const char* filename);

	// TODO, FIXME
	// Warning: Though ugly, it's part of current pipeline ...
	extern double debug_J[];
	extern double debug_outA[];

	void get_jaw_transform(
		const FLAME* flame,
		const float* shape_params, int n_shape_params, // 0-300
		const float* expr_params, int n_expr_params, // 300-400
		const float* pos_params,
		//const float* eyelid_params,
		float hack_scale, float* mat
	);

	void lbs_acc(
		std::vector<Eigen::Vector3f>& out_vertex,
		const std::vector<int>& sel_id,
		const FLAME* flame,
		const float* shape_params, int n_shape_params, // 0-300
		const float* expr_params, int n_expr_params, // 300-400
		const float* pos_params,
		const float* eyelid_params,
		float hack_scale = 1.f
	);

	void lbs(
		std::vector<Eigen::Vector3f>& out_vertex,
		const FLAME* flame,
		const float* shape_params, int n_shape_params, // 0-300
		const float* expr_params, int n_expr_params, // 300-400
		const float* pos_params,
		const float* eyelid_params = nullptr,
		float hack_scale = 1.f
	);

	// NOTE Rely on debug_outA
	void debug_get_jaw_transform(float * mat4x4);

	Eigen::Matrix3f norm_matrix(const Eigen::Matrix3f& mat);

	template<typename T>
	T length3(const T* x) {
		return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
	}

	template<typename T>
	T dot3(const T* x,const T*y) {
		return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
	}

	template<typename T>
	void cross(const T* x, const T* y, T* z) {
		z[0] = x[1] * y[2] - x[2] * y[1];
		z[1] = x[2] * y[0] - x[0] * y[2];
		z[2] = x[0] * y[1] - x[1] * y[0];
	}

	template<typename T>
	void rotation_6d_to_matrix(
		T* outputs, 
		const T* inputs, int n
	) {
		// inputs nx6
		// outputs nx3x3
		for (int i = 0; i < n; i++) {
			T x[3] = { inputs[6 * i + 0], inputs[6 * i + 1], inputs[6 * i + 2] };
			T y[3] = { inputs[6 * i + 3], inputs[6 * i + 4], inputs[6 * i + 5] };
			T L = length3(x);
			T norm_x[3] = { x[0] / L,	x[1] / L, x[2] / L};
			T dot_x_y = dot3(norm_x, y);
			T norm_y[3] = {
				y[0] - dot_x_y * norm_x[0],
				y[1] - dot_x_y * norm_x[1],
				y[2] - dot_x_y * norm_x[2]
			};
			L = length3(norm_y);
			norm_y[0] /= L;	norm_y[1] /= L; norm_y[2] /= L;
			T z[3];
			cross(norm_x, norm_y, z);
			outputs[9 * i + 0] = norm_x[0]; outputs[9 * i + 1] = norm_x[1]; outputs[9 * i + 2] = norm_x[2];
			outputs[9 * i + 3] = norm_y[0]; outputs[9 * i + 4] = norm_y[1]; outputs[9 * i + 5] = norm_y[2];
			outputs[9 * i + 6] = z[0]; outputs[9 * i + 7] = z[1]; outputs[9 * i + 8] = z[2];
		}
	}

	template<typename T>
	void log_so3(
		T* outputs,
		const T* inputs, int n,
		T eps = 1e-4
	) {
		// inputs nx3x3, orthogonal
		// outputs nx3
		for (int i = 0; i < n; i++) {
			T rot_trace = inputs[9 * i + 0] + inputs[9 * i + 4] + inputs[9 * i + 8];
			T phi_cos = (rot_trace - 1) * 0.5;
			if (phi_cos < -1) phi_cos = -1;
			if (phi_cos > 1) phi_cos = 1;
			T phi = acos(phi_cos);
			T phi_sin = sin(phi);
			T phi_factor;
			if (abs(phi_sin) > 0.5 * eps) {
				phi_factor = phi / (2 * phi_sin);
			}
			else { // avoid div tiny number
				phi_factor = 0.5 + (phi * phi) / 12;
			}
			outputs[3 * i + 0] = phi_factor * (inputs[9 * i + 7] - inputs[9 * i + 5]);
			outputs[3 * i + 1] = phi_factor * (inputs[9 * i + 2] - inputs[9 * i + 6]);
			outputs[3 * i + 2] = phi_factor * (inputs[9 * i + 3] - inputs[9 * i + 1]);
		}		
	}

	template<typename T>
	void exp_so3( 
		T* outputs,
		const T* inputs, int n
	) {
		// inputs nx3
		// outputs nx3x3
		for (int i = 0; i < n; i++) {
			T rx, ry, rz;
			rx = inputs[3 * i];
			ry = inputs[3 * i + 1];
			rz = inputs[3 * i + 2];
			T angle = sqrt(rx * rx + ry * ry + rz * rz);
			T rot_dir[3];
			if (angle == 0) {
				rot_dir[0] = 1;	rot_dir[1] = rot_dir[2] = 0;
			}else {
				rot_dir[0] = rx / angle;
				rot_dir[1] = ry / angle;
				rot_dir[2] = rz / angle;
			}
			rx = rot_dir[0];
			ry = rot_dir[1];
			rz = rot_dir[2];
			T cos_angle = cos(angle);
			T sin_angle = sin(angle);
			T* l_outputs = &outputs[9 * i];
			l_outputs[0] = 1 - (1 - cos_angle) * (ry * ry + rz * rz);
			l_outputs[1] = -sin_angle * rz + (1 - cos_angle) * rx * ry;
			l_outputs[2] = sin_angle * ry + (1 - cos_angle) * rx * rz;

			l_outputs[3] = sin_angle * rz + (1 - cos_angle) * rx * ry;
			l_outputs[4] = 1 - (1 - cos_angle) * (rz * rz + rx * rx);
			l_outputs[5] = -sin_angle * rx + (1 - cos_angle) * rz * ry;

			l_outputs[6] = -sin_angle * ry + (1-cos_angle) * rx * rz;
			l_outputs[7] = sin_angle * rx + (1-cos_angle) * rz * ry;
			l_outputs[8] = 1 - (1-cos_angle) * (ry * ry + rx * rx);
		}
	}

	template<typename T>
	void mat4x4_apply(const T* mat, const T* vec3, T* out_vec3) {
		out_vec3[0] = mat[0] * vec3[0] + mat[1] * vec3[1] + mat[2] * vec3[2] + mat[3];
		out_vec3[1] = mat[4] * vec3[0] + mat[5] * vec3[1] + mat[6] * vec3[2] + mat[7];
		out_vec3[2] = mat[8] * vec3[0] + mat[9] * vec3[1] + mat[10] * vec3[2] + mat[11];
	}

	template<typename T>
	void mat4x4_applyrot(const T* mat, const T* vec3, T* out_vec3) {
		out_vec3[0] = mat[0] * vec3[0] + mat[1] * vec3[1] + mat[2] * vec3[2];
		out_vec3[1] = mat[4] * vec3[0] + mat[5] * vec3[1] + mat[6] * vec3[2];
		out_vec3[2] = mat[8] * vec3[0] + mat[9] * vec3[1] + mat[10] * vec3[2];
	}

	template<typename T>
	void matrix_mult4x4(T* results, const T* mat1, const T* mat2) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				T ans = 0;
				for (int p = 0; p < 4; p++) {
					ans += mat1[i * 4 + p] * mat2[p * 4 + j];
				}
				results[i * 4 + j] = ans;
			}
		}
	}

	template<typename T>
	void merge_Rt(T* results, const T* R, const T* t, int n){
		// results n x 4 x 4
		// R = n x 3 x 3, t = n x 3
		for (int i = 0; i < n; i++) {
			results[i * 16 + 0] = R[i * 9 + 0]; results[i * 16 + 1] = R[i * 9 + 1]; results[i * 16 + 2] = R[i * 9 + 2]; results[i * 16 + 3] = t[i * 3 + 0];
			results[i * 16 + 4] = R[i * 9 + 3]; results[i * 16 + 5] = R[i * 9 + 4]; results[i * 16 + 6] = R[i * 9 + 5]; results[i * 16 + 7] = t[i * 3 + 1];
			results[i * 16 + 8] = R[i * 9 + 6]; results[i * 16 + 9] = R[i * 9 + 7]; results[i * 16 + 10] = R[i * 9 + 8]; results[i * 16 + 11] = t[i * 3 + 2];
			results[i * 16 + 12] = results[i * 16 + 13] = results[i * 16 + 14] = 0; results[i * 16 + 15] = 1;
		}	
	}

	template<typename T>
	void compute_blendshape_and_index(
		T* results, const T* params, const T* mean, const T* offset, int* index,
		int b, int n, int k, int n_index, int pos_dim = 3
	) {
		// results b x n' x pos_dim
		// params b x k*
		// offset  n x pos_dim x k*
		// mean[optional] b x n x pos_dim
		// index sel partial results in dimension `n`, reduce dimension => n'
		T* buffer = new T[pos_dim];
		for (int i_batch = 0; i_batch < b; i_batch++) {
			for (int i_idx = 0; i_idx < n_index; i_idx++) {
				int i_n = index[i_idx];
				// mean
				if (mean == nullptr) {
					for (int j = 0; j < pos_dim; j++)
						buffer[j] = 0;
				}
				else {
					for (int j = 0; j < pos_dim; j++) {
						buffer[j] = mean[(i_batch * n + i_n) * pos_dim + j];
					}
				}
				// sum				
				for (int i_k = 0; i_k < k; i_k++) {
					for (int j = 0; j < pos_dim; j++) {
						buffer[j] += params[i_batch * k + i_k] * offset[(i_n * pos_dim + j) * k + i_k];
					}
				}
				// write back
				for (int j = 0; j < pos_dim; j++) {
					results[(i_batch * n_index + i_idx) * pos_dim + j] = buffer[j];
				}
			}
		}	
	}

	template<typename T>
	void compute_blendshape( 
		T * results, const T * params, const T * mean, const T * offset,
		int b, int n, int k, int pos_dim = 3
	) {
		// results b x n x pos_dim
		// params b x k*
		// offset  n x pos_dim x k*
		// mean[optional]  b x n x pos_dim
		T * buffer = new T[pos_dim];
		for (int i_batch = 0; i_batch < b; i_batch++) {
			for (int i_n = 0; i_n < n; i_n++) {
				// mean
				if (mean == nullptr) {
					for (int j = 0; j < pos_dim; j++)
						buffer[j] = 0;
				}
				else {
					for (int j = 0; j < pos_dim; j++) {
						buffer[j] = mean[(i_batch * n + i_n) * pos_dim + j];
					}
				}
				// sum				
				for (int i_k = 0; i_k < k; i_k++) {
					for (int j = 0; j < pos_dim; j++) {
						buffer[j] += params[i_batch * k + i_k] * offset[(i_n * pos_dim + j) * k + i_k];
					}
				}
				// write back
				for (int j = 0; j < pos_dim; j++) {
					results[(i_batch * n + i_n) * pos_dim + j] = buffer[j];
				}
			}
		}
		delete[] buffer;
	}

	//////////////////// Common helper tools

	bool is_root_path(const char* filename);
	std::string cat_path(const char* path, const char* file);
	std::string to_lower(const std::string& s);

	// "D:/File/1.txt" -> "D:/File","1.txt"
	void path_split(const char* filename, char* path, char* file);

	void createDirectoryRecursively(const std::string& directory);
	bool isValidFile(const char* filename);
	// flag = 0, all
	// flag = 1, directory
	// flag = 2, 
	std::vector<std::string> getFilenameInDirectory(const char* dirname, int flag = 0);

}