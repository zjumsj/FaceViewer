#pragma once
#include <functional>

namespace cuGaussianSplatting {

	// GL object mapping to CUDA

	static const int MAX_RESOURCES = 128;
	static const int MAX_STREAMS = 256;

	enum REGISTER_FLAG {
		NONE, READONLY, WRITEDISCARD, SURFACELOADSTORE, TEXTUREGATHER
	};

	// [gl_enum] 
	// target must match the type of the object,
	// and must be one of GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY,
	// or GL_RENDERBUFFER.
	void RegisterGLRenderBuffersOrTextures(unsigned int object_id, unsigned int gl_enum, REGISTER_FLAG flag = NONE, unsigned int unit_idx = 0);
	void RegisterGLBuffers(unsigned int object_id, REGISTER_FLAG flag = NONE, unsigned int unit_idx = 0);
	void UnregisterGLResources(unsigned int unit_idx = 0);

	void GLMappedPointer(void** ptr, size_t& bytes, unsigned int unit_idx = 0);
	void GLUnmapPointer(unsigned int unit_idx = 0);

	//void GenCudaEvents(unsigned int start_resource_idx, int n);
	//void DeleteCudaEvents(unsigned int start_resouce_idx, int n);

	int CreateStreams(int N);
	int GetStreamNumber();
	void DestroyStreams();

	void SetCudaHeapSize(size_t n);

	void CheckCudaError(int i);

	std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S); 

	void HostToDeviceWrapper(void* tar, const void* src, size_t n_bytes);
	void DeviceToHostWrapper(void* tar, const void* src, size_t n_bytes);
	void DeviceToDeviceWrapper(void* tar, const void* src, size_t n_bytes);
	
	// easy to use data structure, like cuGLTest


	///////////////////////////
	// old pipeline

	struct GPoints {
		
		int n_points = 0, l_sh = 0;
		int color_channel = 3;
		int n_scale_elems = 0, n_rot_elems = 0;

		float* pos_cuda = 0;

		float* color_cuda = 0;
		float* shs_cuda = 0;
		
		float* opacity_cuda = 0;

		float* normal_cuda = 0;
		float* rot_cuda = 0;
		float* scale_cuda = 0;

		// extra+
		float* transfer_cuda = 0;

		// extra
		int* rect_cuda = 0;
		float* bkg_cuda = 0;
		//float* buffer_cuda = 0;

		// create cuda array from cpu array
		GPoints();
		
		void initParams(int n_points, int color_channel = 3, int l_sh = 3, int n_scale_elems = 3, int n_rot_elems = 4);
		void allocateData(
			bool pos = true, bool color = false, bool shs = true,
			bool opacity = true, bool normal = false, bool rot = true,
			bool scale = true
		);
		void allocateCopyData(
			float* pos, float* color = nullptr, float* shs = nullptr,
			float* opacity = nullptr, float* normal = nullptr,
			float* rot = nullptr, float* scale = nullptr
		);
		void copyData(
			float * pos, float * color=nullptr, float * shs=nullptr,
			float * opacity = nullptr, float * normal = nullptr,
			float * rot = nullptr, float * scale = nullptr
		);
		void fillColor(float* color3);
		void fillColorWithAnisotropy(float max_rate);
		void fillColorWithDepth(float* modelview);
		static void ColorDepth(float* color, int width, int height, float k, float b, bool inverse);
		void debug_dump(float ** x, int & n, int type_);
		void copyOnlyPos(float* pos);
		void allocateTransferData();
		void copyTransferData(float * transfer);
		void allocateRectData();
		void allocateBkgData(bool white = false, float * replace = nullptr);
		void Destroy();
		~GPoints();
	};

	struct GPoints_basis {

		int n_points = 0, l_sh = 0;
		int color_channel = 3;
		int n_scale_elems = 0, n_rot_elems = 0;
		int n_basis = 0;

		float* pos_t_cuda = 0;
		float* color_t_cuda = 0;
		float* shs_t_cuda = 0;
		float* opacity_t_cuda = 0;
		//float* normal_cuda = 0;
		float* rot_t_cuda = 0;
		float* scale_t_cuda = 0;

		void initParams(int n_points, int n_basis, int color_channel = 3, int l_sh = 3, int n_scale_elems = 3, int n_rot_elems = 4);
		void copyData(
			float* pos_t, float* color_t,
			float* shs_t, float* opacity_t,
			float* rot_t, float* scale_t
		);
		void debug_dump(float** x, int& n, int type_);
		void Destroy();
		~GPoints_basis();
	};

	void CompositeBasis2(
		const float* params,
		const GPoints* mean,
		const GPoints_basis* basis,
		GPoints* result,
		int n_compute_dim,
		bool enable_rot_geom = true,
		bool enable_rot_sh = true
	);

	void CompositeBasis(
		const float * params,
		const GPoints* mean,
		const GPoints_basis * basis,
		GPoints* result,
		int n_compute_dim
	);

	///////////////////////////
	// new pipeline

	struct cuGaussianBasic {

		int n_points = 0, l_sh = 0;
		int color_channel = 3;
		int n_scale_elems = 0, n_rot_elems = 0;

		float* d_pos = 0;
		float* d_rot = 0;
		float* d_scale = 0;
		float* d_opacity = 0;
		float* d_shs = 0;

		float* d_frameRt_NR = 0;  // 16 + 16 + 16 float		

		inline int get_elem_sh() { return (l_sh + 1) * (l_sh + 1) * color_channel; }

		cuGaussianBasic();
		~cuGaussianBasic();
		void initParams(int n_points, int color_channel = 3, int l_sh = 3, int n_scale_elems = 3, int n_rot_elems = 4);
		void Destroy();
		void AllocateData();
		void CopyData(
			float* pos, float* rot, float* scale,
			float* opacity, float* shs
		);
		void allocateTransferData(); 
		void copyTransferData(const float* frameRt, const float * frameNR);
	};


	struct cuGaussianFace : public cuGaussianBasic {

		int n_basis = 0;

		float* d_xyz_t = 0;
		float* d_rot_t = 0;
		float* d_scale_t = 0;
		float* d_opacity_t = 0;
		float* d_shs_t = 0;

		float* d_rot_d = 0; // optional
		float* d_pos_t = 0;
		float* d_W = 0;
		float* d_eyelid = 0;

		float* d_node_transfer = 0;
		mutable float* d_buffer = 0; mutable size_t d_buffer_size = 0;

		bool use_deform_rot = false;
		bool use_deform_rot_sh = false;

		bool use_trans_rot = false;
		bool use_trans_rot_sh = false;

		bool force_50 = false;

		cuGaussianFace();
		~cuGaussianFace();
		inline void setBasis(int n) { n_basis = n; }
		void Destroy();
		void AllocateData(bool alloc_rot_d=true);
		void CopyDataEx(
			float* xyz_t, float* rot_t, float* scale_t, float* opacity_t, float* shs_t,
			float* rot_d, float* pos_t, float* W, float* eyelid
		);
		void allocateNodeTransfer();
		void copyNodeTransfer(const float* node_transfer); // 5*4*4

		// FaceUnity ARKit blendshape
		void AllocateDataSimple();

	};

	void matrix_to_quaternion(float* outq, const float* mat);
	void ScaleActivation(
		const float * src, float * tar, int n_elements, int tar_offset = 0
	);
	void OpacityActivation(
		const float * src, float * tar, int n_elements, int tar_offset = 0
	);
	void RotActivation(
		const float * src, float * tar, int n_elements, int tar_offset = 0
	);

	// copy pos, rot, sh
	void SimpleTransferBasic(
		const cuGaussianBasic * basic,
		GPoints* result,
		bool enable_rot, bool enable_rot_sh,
		int tar_offset = 0
	);

	void CompositeNewPipev1(
		const float* params,
		const float* node_transfer,
		const cuGaussianFace* face,
		GPoints* result,
		int n_compute_dim,
		bool enable_deform_rot, bool enable_deform_rot_sh,
		bool enable_trans_rot, bool enable_trans_rot_sh,
		int tar_offset = 0, bool force_50 = false
	);

	void CompositeFUPipev1(
		const float* params,
		const cuGaussianFace * face,
		GPoints * result,
		int n_compute_dim,
		int tar_offset = 0
	);
	
	///////////////////////////
	

	struct Transforms {

		// Col Major

		float mvpMatrix[16];
		float mvMatrix[16];
		float c2wMatrix[16];
		float projMatrix[16];
		int vpParams[4];// viewport(x,y,w,h)

		void* matrices = nullptr;

		bool Initialize();
		bool CopyData(); // call me after fill all matrices...
		void Destroy();
		~Transforms();

	};


	struct Canvas {
		int width = 0;
		int height = 0;
		int channel = 0;
		void* canvas = nullptr;

		// 3xHxW
		bool Create(int width, int height, int channel = 3);
		//bool Fill(float v, int x0, int x_size, int y0, int y_size, int c0, int c_size);
		//bool Fill(float v);
		void Destroy();
		~Canvas();
	};

	struct cuBuffer{

		int n_byte = 0;
		void* params = 0;

		bool Initialize(int n_byte);
		void Destroy();
		void Copy(void* src_data, int n_byte);
		void CopyBack(void* tar_data, int n_byte);
		~cuBuffer();

	};

	struct cuBlendTex {
		unsigned int TexID;
		int width, height, c,k;
		void * mean; // HxWxc
		void * basis; // HxWxcxk
		void * params; // k

		static const int register_slot = 7;
		
		cuBlendTex();
		bool Initialize(int w, int h, int channel, int n_basis);
		// the input tex is in CV coord, you may need to transfer to GL coord
		bool CopyData(const float* basis_, const float* mean_ = nullptr); // TODO, add flip later
		bool CopyDataD(const double* basis_, const double* mean_ = nullptr, bool flip=true);
		bool Blend(const float* params_, int n);
		void Destroy();
		~cuBlendTex();
	private:
		bool has_mean;
	};
}