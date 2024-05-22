#pragma once

#include "io.h"

//#include "../cnpy/cnpy.h"
#include "cnpy/cnpy.h"

#include <string>
#include <map>

// Shallow package for simple use
// Process oriented
namespace gaussian_splatting {

	struct gl_buffer_object {
			
		void Init();
		bool Allocate(int n_bytes, void * ptr = nullptr, unsigned int type_ = 0);
		bool Subdata(int n_bytes, void* ptr);
		void Release();
		gl_buffer_object();
		~gl_buffer_object();

		unsigned int bufferID;
	};

	struct gaussian_points_coord {

		std::vector<FaceCoord> point_coord;
		void load(const char* filename);

	};

	struct gaussian_points {

		std::vector<Pos> pos;
		std::vector<Rot> rot;
		std::vector<Scale> scale;
		std::vector<float> opacity;
		std::vector<SHs<3>> shs;

		std::vector<Eigen::Vector3f> normal;
		std::vector<Eigen::Vector3f> color;

		Eigen::Vector3f minn;
		Eigen::Vector3f maxx;

		void load(const char* filename);
	};

	struct gaussian_points_basis {

		//int n_basis = 0;

		//cnpy::NpyArray pos_mean; 
		//cnpy::NpyArray shs_mean;
		//cnpy::NpyArray scale_mean;
		//cnpy::NpyArray rot_mean;
		//cnpy::NpyArray opacity_mean;

		cnpy::NpyArray pos_t; 
		cnpy::NpyArray shs_t;
		cnpy::NpyArray scale_t;
		cnpy::NpyArray rot_t;
		cnpy::NpyArray opacity_t;

		void load(const char* filename);
		int getK();
	};

	//void hack_remove_points_on_triangle() {
	//
	//}

	struct transform_v1 {

		std::vector<int> back_head_vertex;
		std::vector<Eigen::Vector3f> expr_back_head_vertex;
		std::vector<Eigen::Vector3f> neutral_back_head_vertex;

		// global rigid transform
		float frameR[16]; // 4x4
		float frameNR[9]; // 3x3

		// jaw transform
		float rjawTrans[16]; // 4x4
		float rjawTransN[9]; // 3x3

		float A0[5 * 4 * 4];
		float A[5 * 4 * 4];

		std::vector<Eigen::Vector3f> neutral_pos;

		void ComputeJawTransFromA();
		void ComputeBackHeadVertex(
			const std::vector<Eigen::Vector3f>& vertex,
			const std::vector<Eigen::Vector3f>& vertex0
		);
		void ComputeBackHeadVertexDirect(
			const std::vector<Eigen::Vector3f>& vertex,
			const std::vector<Eigen::Vector3f>& vertex0
		);
		void ComputeRelA(float* T) const;
	private:
		void get_jaw_transform (const float * matA, float* mat4x4) const;
		
	};

	struct local_transfer {

		std::vector<Eigen::Vector3f> pos_offset;
		std::vector<Eigen::Vector3f> neutral_pos;
		std::vector<Eigen::Matrix3f> local_frame;
		std::vector<Eigen::Matrix3f> faceR;
		std::vector<Eigen::Vector3f> facet;
		std::vector<Eigen::Matrix3f> faceNR;
		std::vector<Eigen::Matrix3f> pointNR;

	};

	struct simple_draw_info {	
		int normal_attrib = -1; int normal_offset = 3;
		int color_attrib = -1; int color_offset = 3;
		int n_stride = 6;
		int n_call;
	};
	// m_primitive = 0 POINTS
	// m_primitive = 1 LINES
	void simple_draw(unsigned int bufferID, const int m_primitive, const simple_draw_info& info);

	// 3DMM FLAME model
	template<typename T>
	struct SparseMatrix {

		struct SparseMatrixItem{
			int id; T v;
		};
		struct SparseMatrixHeader {
			int n_items;
			SparseMatrixItem* items = nullptr;
		};
		
		int n_row = 0;
		SparseMatrixHeader * ptr = nullptr;
		
		void from_dense(const T* src, int n_row, int n_col) {
			this->n_row = n_row;
			if (ptr) clear();
			if (n_row > 0) {
				ptr = new SparseMatrixHeader[n_row];
				for (int i_row = 0; i_row < n_row; i_row++) {
					int i_col_count = 0;
					int i_col = 0;
					while (i_col < n_col) {
						if (src[i_row * n_col + i_col] != 0) {
							i_col_count++;
						}
						i_col++;
					}
					ptr[i_row].n_items = i_col_count;
					if (i_col_count) {
						ptr[i_row].items = new SparseMatrixItem[i_col_count];
						i_col = 0;
						i_col_count = 0;
						while (i_col < n_col) {
							T v = src[i_row * n_col + i_col];
							if (v != 0) {
								ptr[i_row].items[i_col_count].id = i_col;
								ptr[i_row].items[i_col_count].v = v;
								i_col_count++;
							}
							i_col++;
						}
					}
				}
			}
		}

		// self = m x idim, 
		// inputs = idim x n  is sparse in `idim` dimension
		// inputs_index, non zero index of inputs in `idim` dimension
		// s = non zero terms
		// outputs = m x n
		void op1(
			T* outputs, const T* inputs, const int* inputs_index,
			int s, int m, /*int i_dim,*/ int n
		) const {
			T* buff = new T[n];
			for (int i_row = 0; i_row < m; i_row++) {
				for (int j = 0; j < n; j++) {
					buff[j] = 0;
				}
				const SparseMatrixHeader* head = this->ptr + i_row;
				for (int is = 0; is < head->n_items; is++) {
					int index0 = head->items[is].id; //
					// TODO FIXME: bruteforce is stupid !
					for (int iss = 0; iss < s; iss++) {
						int index1 = inputs_index[iss];
						if (index0 == index1) { // find match
							T v = head->items[is].v;
							for (int j = 0; j < n; j++) {
								buff[j] += v * inputs[iss * n + j];
							}
							break;
						}
					}
				}
				//int ptr0 = 0;
				//int ptr1 = 0;
				//while (ptr0 < s && ptr1 < head->n_items) {
				//	int index0 = inputs_index[ptr0]; // go through input matrix
				//	int index1 = head->items[ptr1].id; // go through self
				//	if (index0 == index1) {
				//		T v = head->items[ptr1].v;
				//		for (int j = 0; j < n; j++) {
				//			buff[j] += v * inputs[ptr0 * n + j];
				//		}
				//		ptr0++; ptr1++;
				//	}
				//	else if (index0 > index1) {
				//		ptr1++;
				//	}
				//	else { // index0 < index1
				//		ptr0++;
				//	}
				//}
				for (int j = 0; j < n; j++) {
					outputs[i_row * n + j] = buff[j];
				}
			}
			delete[] buff;
		}

		//  self = m x idim , inputs = b x idim x n -> outputs = b x m x n
		void apply(
			T* outputs, const T* inputs,
			int b, int m, int i_dim, int n
		) const {
			T* buff = new T[n];
			for (int i_batch = 0; i_batch < b; i_batch++) {
				for (int i_row = 0; i_row < m; i_row++) {
					const SparseMatrixHeader* head = this->ptr + i_row;
					for (int j = 0; j < n; j++) {
						buff[j] = 0;
					}
					for (int i = 0; i < head->n_items; i++) {
						int id = head->items[i].id;
						T v = head->items[i].v;
						for (int j = 0; j < n; j++) {
							buff[j] += v * inputs[(i_batch * i_dim + id) * n + j];
						}
					}
					for (int j = 0; j < n; j++) {
						outputs[(i_batch * m + i_row) * n + j] = buff[j];
					}
				}
			}
			delete[] buff;
		}
		cnpy::NpyArray apply(const cnpy::NpyArray& inputs) const {
			// 5(m)x5023(i), 1(b)x5023(i)x3(n)
			int b = inputs.shape(0);
			int dim_i = inputs.shape(1);
			int m = n_row;
			int n = inputs.shape(2);			
			cnpy::NpyArray output = cnpy::NpyArray({ b,m,n }, sizeof(T), false);
			const T* input_ptr = inputs.data<T>();
			T* output_ptr = output.data<T>();
			apply(output_ptr, input_ptr, b, m, dim_i, n);
		}
		void clear() {
			if (ptr) {
				for (int i_row = 0; i_row < n_row; i_row++) {
					SparseMatrixHeader* head = &ptr[i_row];
					if (head->items) delete[] head->items;
				}
				delete[] ptr; ptr = nullptr;
			}
		}
		~SparseMatrix() {
			clear();
		}
	};


	struct FLAME {

		std::string bs_style;
		std::string bs_type;

		cnpy::NpyArray J; // (5,3)
		cnpy::NpyArray J_regressor; // should sparse, 5, n_vert
		SparseMatrix<double> J_regressor_;

		cnpy::NpyArray f; // n_face x 3, uint32
		cnpy::NpyArray kintree_table; // (2,5) int64

		cnpy::NpyArray posedirs; // n_vert * 3 * 36
		cnpy::NpyArray shapedirs; // n_vert * 3 * 400
		cnpy::NpyArray v_template; // n_vert * 3
		cnpy::NpyArray weights; // n_vert * 5

		// extra eyelid data
		cnpy::NpyArray * l_eyelid;
		cnpy::NpyArray * r_eyelid;
		// debug
		cnpy::NpyArray* shape_mean;

		void load(const char* filename);
		void compute_shape_mean();
		FLAME();
		~FLAME();
	};

	struct FLAMETex {

		cnpy::NpyArray mean; // (512,512,3)
		cnpy::NpyArray tex_dir; //(512,512,3,200)
		cnpy::NpyArray vt; // (vert,2) UV per vertex?
		cnpy::NpyArray ft; // (face,3) uint32

		void load(const char* filename);
	};

	struct FaceTexParams {

		static const int n_params_tex = 200;
		float params_tex[n_params_tex] = { 0 };
		float ui_params_tex[n_params_tex] = { -1 };

		void resetTexParams();
		inline void SetIgnoreOnce() { ignore_diff = true; }
		bool FindDiff();

	private:
		void Copy();
		bool ignore_diff = false;
	};

	struct FaceParams {

		static const int n_params_shape = 300;
		static const int n_params_expr = 100;
		static const int n_params_pose = 15;
		static const int n_params_eyelid = 2;		

		float params_shape[n_params_shape] = { 0 };
		float params_expr[n_params_expr] = { 0 };
		float params_pose[n_params_pose] = { 0 };
		float params_eyelid[n_params_eyelid] = { 0 };

		// for display
		float ui_params_shape[n_params_shape] = { -1 };
		float ui_params_expr[n_params_expr] = { -1 };
		float ui_params_pose[n_params_pose] = { -1 };
		float ui_params_eyelid[n_params_eyelid] = { -1 };

		void resetShapeParams();
		void resetExprParams();
		void resetPoseParams();
		void resetEyelidParams();
		
		inline void SetIgnoreOnce() { ignore_diff = true; }
		bool FindDiff(); 

	private:
		void Copy();
		bool ignore_diff = false;
	};

	struct TorchFrame {

		//std::vector<float> exp;
		//std::vector<float> shape;
		//std::vector<float> tex;
		//std::vector<float> sh;
		//std::vector<float> eyes;
		//std::vector<float> eyelids;

		//std::vector<float> R;
		//std::vector<float> t;
		//std::vector<float> K;

		std::map<std::string, std::vector<float>> params;

		int64_t img_size[2];
		std::string frame_id;
		int64_t global_step;

		void load(const char* filename);
	};

	struct FUFrame {

		static const int n_params_shape = 231;
		static const int n_params_expr = 51;

		bool is_valid = false;
		int id;
		float params_shape[n_params_shape];
		float params_expr[n_params_expr];
		float params_rot[4]; // quaternion x,y,z,w
		float params_trans[3];
		float params_eye_rot[8]; // left,right quaternion x,y,z,w
		

		void get_camera(float * w2c) const; // Col-major 4x4 matrix
		void get_eyemat(int is_right, float* mat3x3, bool transpose = true) const; // Col-major 3x3 matrix
	};

	void load(const char* filename, std::vector<FUFrame>& frames);

	struct SimpleCameraFrame{
		float c2w[12]; 
		// in col major
		// 0 3 6 9
		// 1 4 7 10
		// 2 5 8 11
	};

	struct CameraVisualizer {
		std::vector<SimpleCameraFrame> v;
		float* vertex_buffer = nullptr; //(p0,p1,p2,...)
		float* frame_buffer = nullptr; //(0,x,0,y,0,z)
		// you should update manually
		void update(float length = 1.f);
		~CameraVisualizer();
		bool load(const char* filename);
		bool load(const TorchFrame* ptr, int n);
		bool load(const std::vector<FUFrame>& frames);
		void scale(float sx, float sy, float sz);
		inline int size() const { return v.size(); }
	};

	void draw_camera_pos(const CameraVisualizer& p, int render_mode = 0, int offset=0, int size=-1);
	void draw_camera_frame(const CameraVisualizer& p, int sel = -1);

	struct SimpleTexture {
		unsigned int TexID;
		int width, height, c;
		void* data;
		SimpleTexture();
		//void buildTexture(void * data = nullptr);
		void destroyTexture();
		~SimpleTexture();
	};

}