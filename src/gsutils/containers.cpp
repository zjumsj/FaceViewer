#include "containers.h"

#include "operators.h"

#include <GL/glew.h>

#include <stdexcept>

extern void GetError();

namespace gaussian_splatting {

	gl_buffer_object::gl_buffer_object() :bufferID(0) {}
	gl_buffer_object::~gl_buffer_object() { Release(); }
	
	void gl_buffer_object::Init() {
		if (bufferID == 0) {
			glGenBuffers(1, &bufferID);
		}
	}

	bool gl_buffer_object::Subdata(int n_bytes, void* ptr) {
		if (bufferID) {
			glBindBuffer(GL_ARRAY_BUFFER, bufferID);
			glBufferSubData(GL_ARRAY_BUFFER, 0, n_bytes, ptr);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return true;
		}
		return false;
	}

	bool gl_buffer_object::Allocate(int n_bytes, void * ptr, unsigned int type_) {
		if (bufferID) {
			glBindBuffer(GL_ARRAY_BUFFER, bufferID);
			if (type_ == 0) type_ = GL_STATIC_DRAW;
			glBufferData(GL_ARRAY_BUFFER, n_bytes, ptr, type_);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			return true;
		}
		return false;
	}

	void gl_buffer_object::Release() {
		if (bufferID) {
			glDeleteBuffers(1, &bufferID);
			bufferID = 0;
		}
	}

	///////////////

	void transform_v1::get_jaw_transform(const float * matA, float* mat4x4) const {
		for (int i = 0; i < 16; i++) {
			mat4x4[i] = matA[i] * 1.30787707e-02 + \
				matA[16 + i] * 4.26467828e-02 + \
				matA[32 + i] * 9.44269622e-01;
		}
	}

	void transform_v1::ComputeJawTransFromA() {
		//-----------------------
		// require: A0, A
		// output: rjawTrans, rjawTransN
		//-----------------------

		float jawTrans[16];
		float jawTrans0[16];
		get_jaw_transform(A, jawTrans);
		get_jaw_transform(A0, jawTrans0);
		for (int i = 0; i < 16; i++) {
			rjawTrans[i] = jawTrans[i] - jawTrans0[i];
		}
		// rjawTrans = torch.eye(4,device=jawTrans.device) + jawTrans - jawTrans0
		// add identity
		rjawTrans[0] += 1.f; rjawTrans[5] += 1.f; rjawTrans[10] += 1.f;	rjawTrans[15] += 1.f;

		Eigen::Matrix3f R_of_jawTrans;
		R_of_jawTrans << rjawTrans[0], rjawTrans[1], rjawTrans[2],
			rjawTrans[4], rjawTrans[5], rjawTrans[6],
			rjawTrans[8], rjawTrans[9], rjawTrans[10];
		Eigen::JacobiSVD<Eigen::Matrix3f> svd;
		//svd.compute(R_of_jawTrans, Eigen::ComputeThinV | Eigen::ComputeThinU);
		svd.compute(R_of_jawTrans, Eigen::ComputeFullV | Eigen::ComputeFullU);
		// U,S,Vt = svd(R)
		Eigen::Matrix3f NR = svd.matrixU() * svd.matrixV().transpose(); // U * Vt
		if (NR.determinant() < 0.f) {
			throw std::runtime_error("Unexpected numerical problem!");
		}
		rjawTransN[0] = NR(0, 0); rjawTransN[1] = NR(0, 1); rjawTransN[2] = NR(0, 2);
		rjawTransN[3] = NR(1, 0); rjawTransN[4] = NR(1, 1); rjawTransN[5] = NR(1, 2);
		rjawTransN[6] = NR(2, 0); rjawTransN[7] = NR(2, 1); rjawTransN[8] = NR(2, 2);
	}
	
	void transform_v1::ComputeBackHeadVertexDirect(
		const std::vector<Eigen::Vector3f>& vertex,
		const std::vector<Eigen::Vector3f>& vertex0
	) {
		// compute global transfer
		Eigen::Matrix3f R; Eigen::Vector3f t;
		gaussian_splatting::rot_match_back_head( // neutral -> expr
			vertex0, vertex, R, t
		);
		frameR[0] = R(0, 0); frameR[1] = R(0, 1); frameR[2] = R(0, 2); frameR[3] = t[0];
		frameR[4] = R(1, 0); frameR[5] = R(1, 1); frameR[6] = R(1, 2); frameR[7] = t[1];
		frameR[8] = R(2, 0); frameR[9] = R(2, 1); frameR[10] = R(2, 2); frameR[11] = t[2];
		frameR[12] = frameR[13] = frameR[14] = 0.f; frameR[15] = 1.f;

		frameNR[0] = R(0, 0); frameNR[1] = R(0, 1); frameNR[2] = R(0, 2);
		frameNR[3] = R(1, 0); frameNR[4] = R(1, 1); frameNR[5] = R(1, 2);
		frameNR[6] = R(2, 0); frameNR[7] = R(2, 1); frameNR[8] = R(2, 2);
	}

	void transform_v1::ComputeBackHeadVertex(
		const std::vector<Eigen::Vector3f>& vertex,
		const std::vector<Eigen::Vector3f>& vertex0
	) {
		expr_back_head_vertex.clear();
		neutral_back_head_vertex.clear();
		for (int i = 0; i < back_head_vertex.size(); i++) {
			int id = back_head_vertex[i];
			expr_back_head_vertex.push_back(vertex[id]);
			neutral_back_head_vertex.push_back(vertex0[id]);
		}
		// compute global transfer
		Eigen::Matrix3f R; Eigen::Vector3f t;
		gaussian_splatting::rot_match_back_head( // neutral -> expr
			neutral_back_head_vertex, expr_back_head_vertex, R, t
		);
		frameR[0] = R(0, 0); frameR[1] = R(0, 1); frameR[2] = R(0, 2); frameR[3] = t[0];
		frameR[4] = R(1, 0); frameR[5] = R(1, 1); frameR[6] = R(1, 2); frameR[7] = t[1];
		frameR[8] = R(2, 0); frameR[9] = R(2, 1); frameR[10] = R(2, 2); frameR[11] = t[2];
		frameR[12] = frameR[13] = frameR[14] = 0.f; frameR[15] = 1.f;

		frameNR[0] = R(0, 0); frameNR[1] = R(0, 1); frameNR[2] = R(0, 2);
		frameNR[3] = R(1, 0); frameNR[4] = R(1, 1); frameNR[5] = R(1, 2);
		frameNR[6] = R(2, 0); frameNR[7] = R(2, 1); frameNR[8] = R(2, 2);

		// [REMOVE] no need to normalize again !
		//Eigen::JacobiSVD<Eigen::Matrix3f> svd;
		////svd.compute(R, Eigen::ComputeThinV | Eigen::ComputeThinU);
		//svd.compute(R, Eigen::ComputeFullV | Eigen::ComputeFullU);
		//// U,S,Vt = svd(R)
		//Eigen::Matrix3f NR = svd.matrixU() * svd.matrixV().transpose(); // U * Vt
		//frameNR[0] = NR(0, 0); frameNR[1] = NR(0, 1); frameNR[2] = NR(0, 2);
		//frameNR[3] = NR(1, 0); frameNR[4] = NR(1, 1); frameNR[5] = NR(1, 2);
		//frameNR[6] = NR(2, 0); frameNR[7] = NR(2, 1); frameNR[8] = NR(2, 2);		
	}

	void transform_v1::ComputeRelA(float* T) const {
		//-----------------------
		// require: A0, A
		// output: eye + A - A0
		//-----------------------
		const int n_node = 5;
		for (int i = 0; i < n_node; i++) {
			for (int j = 0; j < 4 * 4; j++) {
				T[i * 16 + j] = A[i * 16 + j] - A0[i * 16 + j];
			}
			T[i * 16 + 0] += 1.f;
			T[i * 16 + 5] += 1.f;
			T[i * 16 + 10] += 1.f;
			T[i * 16 + 15] += 1.f;
		}	
	}

	///////////////

	void simple_draw(unsigned int bufferID, const int m_primitive, const simple_draw_info & info) {
		
		glBindBuffer(GL_ARRAY_BUFFER, bufferID);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * info.n_stride, 0);
		if (info.normal_attrib >= 0) {
			glEnableVertexAttribArray(info.normal_attrib);
			glVertexAttribPointer(info.normal_attrib, 3, GL_FLOAT, GL_FALSE, sizeof(float) * info.n_stride, 0);
		}
		if (info.color_attrib >= 0) {
			glEnableVertexAttribArray(info.color_attrib);
			glVertexAttribPointer(info.color_attrib, 3, GL_FLOAT, GL_FALSE, sizeof(float) * info.n_stride, reinterpret_cast<void*>(info.color_offset * sizeof(float)));
		}

		if (m_primitive == 0)
			glDrawArrays(GL_POINTS, 0, info.n_call);
		else if (m_primitive == 1) // you can hack and add more primitives 
			glDrawArrays(GL_LINES, 0, info.n_call);


		glDisableVertexAttribArray(0);
		if (info.normal_attrib >= 0) glDisableVertexAttribArray(info.normal_attrib);
		if (info.color_attrib >= 0) glDisableVertexAttribArray(info.color_attrib);
	
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	/////////////////////////////////

	void gaussian_points_coord::load(const char* filename) {
		
		size_t n_points;
		cnpy::NpyArray npy_face_id = cnpy::npy_load(cat_path(filename, "face_id.npy"));
		cnpy::NpyArray npy_barycentric_coord = cnpy::npy_load(cat_path(filename, "barycentric_coord.npy"));
		n_points = npy_face_id.shape[0];
		assert(npy_barycentric_coord.shape[0] == n_points);
		point_coord.resize(n_points);
		for (int i = 0; i < n_points; i++) {
			FaceCoord& fc = point_coord[i];
			fc.face_id = (int)npy_face_id.data<int64_t>()[i];
			fc.barycentric_coord[0] = npy_barycentric_coord.data<float>()[3 * i + 0];
			fc.barycentric_coord[1] = npy_barycentric_coord.data<float>()[3 * i + 1];
			fc.barycentric_coord[2] = npy_barycentric_coord.data<float>()[3 * i + 2];
		}		
	}

	void gaussian_points::load(const char* filename) {

		size_t n_points;
		size_t channel;
		size_t level;
		size_t total_level; // (3+1)*(3+1)

		if (gaussian_splatting::isValidFile(cat_path(filename, "shs.npy").c_str())) {
			cnpy::NpyArray npy_shs = cnpy::npy_load(cat_path(filename, "shs.npy"));
			// Px3xN X
			// PxNx3
			//assert(npy_shs.shape.size() == 3 && npy_shs.shape[1] == 3);
			assert(npy_shs.shape.size() == 3 && npy_shs.shape[2] == 3);
			n_points = npy_shs.shape[0];
			//channel = npy_shs.shape[1];
			//level = npy_shs.shape[2];
			
			level = npy_shs.shape[1];
			shs.resize(n_points);

			for (int i = 0; i < n_points; i++) {
				SHs<3>& sh = shs[i];
				total_level = sizeof(sh.shs) / sizeof(float) / 3;
				for (int i_l = 0; i_l < total_level * 3; i_l++) {// clear data
					sh.shs[i_l] = 0;
				}
				//for (int i_l = 0; i_l < level; i_l++) { // load
				//	sh.shs[i_l * 3 + 0] = npy_shs.data<float>()[(i * 3 + 0) * total_level + i_l];
				//	sh.shs[i_l * 3 + 1] = npy_shs.data<float>()[(i * 3 + 1) * total_level + i_l];
				//	sh.shs[i_l * 3 + 2] = npy_shs.data<float>()[(i * 3 + 2) * total_level + i_l];
				//}
				for (int i_e = 0; i_e < level * 3; i_e++) {
					sh.shs[i_e] = npy_shs.data<float>()[i * total_level * 3 + i_e];
				}
			}
		}
		else if (gaussian_splatting::isValidFile(cat_path(filename, "features_dc.npy").c_str())) {
			// Px1x3
			// Px(K-1)x3
			// TODO
		}
		else {
			throw std::runtime_error("can not find file!");
		}

		if(gaussian_splatting::isValidFile(cat_path(filename,"pos.npy").c_str()))
		{
			cnpy::NpyArray npy_pos = cnpy::npy_load(cat_path(filename, "pos.npy"));
			n_points = npy_pos.shape[0];
			pos.resize(n_points);
			for (int i = 0; i < n_points; i++) {
				Pos& p = pos[i];
				p.x() = npy_pos.data<float>()[3 * i + 0];
				p.y() = npy_pos.data<float>()[3 * i + 1];
				p.z() = npy_pos.data<float>()[3 * i + 2];
			}
		}

		{
			cnpy::NpyArray npy_rot = cnpy::npy_load(cat_path(filename, "rot.npy"));
			n_points = npy_rot.shape[0];
			rot.resize(n_points);
			for (int i = 0; i < n_points; i++) {
				Rot& r = rot[i];
				r.rot[0] = npy_rot.data<float>()[4 * i + 0];
				r.rot[1] = npy_rot.data<float>()[4 * i + 1];
				r.rot[2] = npy_rot.data<float>()[4 * i + 2];
				r.rot[3] = npy_rot.data<float>()[4 * i + 3];
			}
		}
		{
			cnpy::NpyArray npy_scale = cnpy::npy_load(cat_path(filename, "scale.npy"));
			n_points = npy_scale.shape[0];
			scale.resize(n_points);
			for (int i = 0; i < n_points; i++) {
				Scale& s = scale[i];
				s.scale[0] = npy_scale.data<float>()[3 * i + 0];
				s.scale[1] = npy_scale.data<float>()[3 * i + 1]; 
				s.scale[2] = npy_scale.data<float>()[3 * i + 2]; 
			}
		}
		{
			cnpy::NpyArray npy_opacity = cnpy::npy_load(cat_path(filename, "opacity.npy"));
			n_points = npy_opacity.shape[0];
			opacity.resize(n_points);
			for (int i = 0; i < n_points; i++) {
				float& s = opacity[i];
				s = npy_opacity.data<float>()[i];
			}
		}
	}

	void gaussian_points_basis::load(const char* filename) {

		////pos_mean = cnpy::npy_load(cat_path(filename, "pos_mean.npy"));
		//shs_mean = cnpy::npy_load(cat_path(filename, "shs_mean.npy"));
		//scale_mean = cnpy::npy_load(cat_path(filename, "scale_mean.npy"));
		//rot_mean = cnpy::npy_load(cat_path(filename, "rot_mean.npy"));
		//opacity_mean = cnpy::npy_load(cat_path(filename, "opacity_mean.npy"));

		//pos_t = cnpy::npy_load(cat_path(filename, "pos_t.npy"));
		shs_t = cnpy::npy_load(cat_path(filename, "shs_t.npy"));
		scale_t = cnpy::npy_load(cat_path(filename, "scale_t.npy"));
		rot_t = cnpy::npy_load(cat_path(filename, "rot_t.npy"));
		opacity_t = cnpy::npy_load(cat_path(filename, "opacity_t.npy"));
	}

	int gaussian_points_basis::getK() {
		if (shs_t.shape.size() == 0) throw std::runtime_error("Uninitialized!");
		int n_dim = shs_t.shape.size();
		return shs_t.shape[n_dim - 1];
	}

	/// <summary>
	/// ////////////////////////////
	/// </summary>

	FLAME::FLAME()
		:l_eyelid(0), r_eyelid(0), shape_mean(0)
	{}

	FLAME::~FLAME() {
		if (l_eyelid) delete l_eyelid;
		if (r_eyelid) delete r_eyelid;
		if (shape_mean) delete shape_mean;
	}

	void FLAME::compute_shape_mean() {
		// input: n_vertex x 3 x k
		// output: 3 x k
		uint64_t n_vertex = shapedirs.shape[0];
		uint64_t pos_dim = shapedirs.shape[1];
		uint64_t k = shapedirs.shape[2];
		double * mean = new double[pos_dim];
		if (shape_mean == nullptr) shape_mean = new cnpy::NpyArray();
		*shape_mean = cnpy::NpyArray({ pos_dim,k }, sizeof(double),false);
		for (auto i_k = 0; i_k < k; i_k++) {
			for (auto j = 0; j < pos_dim; j++) {
				mean[j] = 0;
			}
			for (auto i_n = 0; i_n < n_vertex; i_n++) {
				for (auto j = 0; j < pos_dim; j++) {
					mean[j] += shapedirs.data<double>()[((i_n * pos_dim) + j) * k + i_k];
				}
			}
			for (auto j = 0; j < pos_dim; j++) {
				shape_mean->data<double>()[j * k + i_k] = mean[j] / double(n_vertex);
			}
		}
		//int ik = 3;		
		//printf("%e %e %e\n", 
		//	shape_mean->data<double>()[0 * k + ik],
		//	shape_mean->data<double>()[1 * k + ik],
		//	shape_mean->data<double>()[2 * k + ik]);
		//
		//ik = 307;
		//printf("%e %e %e\n",
		//	shape_mean->data<double>()[0 * k + ik],
		//	shape_mean->data<double>()[1 * k + ik],
		//	shape_mean->data<double>()[2 * k + ik]);

		delete[] mean;	
	}


	void FLAME::load(const char* filename) {
		
		J = cnpy::npy_load(cat_path(filename,"J.npy"));
		J_regressor = cnpy::npy_load(cat_path(filename, "J_regressor.npy"));
		// transfer to sparse matrix
		J_regressor_.from_dense(J_regressor.data<double>(), J_regressor.shape[0], J_regressor.shape[1]);

		f = cnpy::npy_load(cat_path(filename, "f.npy"));
		kintree_table = cnpy::npy_load(cat_path(filename, "kintree_table.npy"));

		posedirs = cnpy::npy_load(cat_path(filename, "posedirs.npy"));
		shapedirs = cnpy::npy_load(cat_path(filename, "shapedirs.npy"));
		v_template = cnpy::npy_load(cat_path(filename, "v_template.npy"));
		weights = cnpy::npy_load(cat_path(filename, "weights.npy"));
	
		{
			std::string tmpc;
			tmpc = cat_path(filename, "l_eyelid.npy");
			if (isValidFile(tmpc.c_str())) {
				if(l_eyelid == nullptr)	l_eyelid = new cnpy::NpyArray();
				*l_eyelid = cnpy::npy_load(tmpc);	
				printf("load extra file %s\n", tmpc.c_str());
			}
			tmpc = cat_path(filename, "r_eyelid.npy");
			if (isValidFile(tmpc.c_str())) {
				if (r_eyelid == nullptr) r_eyelid = new cnpy::NpyArray();
				*r_eyelid = cnpy::npy_load(tmpc);
				printf("load extra file %s\n", tmpc.c_str());
			}
		}

		if (false) { // debug
			printf("J %f %f\n", *J.data<double>(), J.data<double>()[1]);
			printf("f %d\n", *f.data<uint32_t>());
			printf("kintree_table %d\n", *kintree_table.data<uint32_t>());
			printf("posedirs %f\n", *posedirs.data<double>());
			printf("shapedirs %f\n", *shapedirs.data<double>());
			printf("v_template %f\n", *v_template.data<double>());
			printf("weights %f\n", *weights.data<double>());
		}
	}

	void FLAMETex::load(const char* filename) {
		
		mean = cnpy::npy_load(cat_path(filename, "mean.npy"));
		tex_dir = cnpy::npy_load(cat_path(filename, "tex_dir.npy"));
		vt = cnpy::npy_load(cat_path(filename, "vt.npy"));
		ft = cnpy::npy_load(cat_path(filename, "ft.npy"));

		if (false) { // debug
			printf("mean %f\n", *mean.data<double>());
			printf("tex_dir %f\n", *tex_dir.data<double>());
			printf("vg %f\n", *vt.data<double>());
			printf("ft %d\n", *ft.data<uint32_t>());
		}
	}

	
	/////////////////

	void FaceTexParams::resetTexParams() {
		for (int i = 0; i < n_params_tex; i++) {
			params_tex[i] = 0.f;
		}
	}

	bool FaceTexParams::FindDiff() {
		if (ignore_diff) {
			Copy();
			ignore_diff = false;
			return false;
		}
		else {
			bool is_diff = false;
			if (!is_diff) {
				for (int i = 0; i < n_params_tex; i++) {
					if (params_tex[i] != ui_params_tex[i]) {
						is_diff = true; break;
					}
				}
			}
			if (is_diff) Copy();
			return is_diff;
		}
	}

	void FaceTexParams::Copy() {
		for (int i = 0; i < n_params_tex; i++) {
			ui_params_tex[i] = params_tex[i];
		}
	}

	/////////////////

	void FaceParams::resetShapeParams(){
		for (int i = 0; i < n_params_shape; i++) {
			params_shape[i] = 0.f;
		}	
	}
	void FaceParams::resetExprParams() {
		for (int i = 0; i < n_params_expr; i++) {
			params_expr[i] = 0.f;
		}
	}
	void FaceParams::resetPoseParams() {
		for (int i = 0; i < n_params_pose; i++) {
			params_pose[i] = 0.f;
		}
	}
	void FaceParams::resetEyelidParams() {
		for (int i = 0; i < n_params_eyelid; i++) {
			params_eyelid[i] = 0.f;
		}
	}
	bool FaceParams::FindDiff() {
		if (ignore_diff) {
			Copy();
			ignore_diff = false;
			return false;
		}
		else {
			bool is_diff = false;
			if (!is_diff) {
				for (int i = 0; i < n_params_shape; i++) {
					if (params_shape[i] != ui_params_shape[i]) {
						is_diff = true; break;
					}
				}
			}
			if (!is_diff) {
				for (int i = 0; i < n_params_expr; i++) {
					if (params_expr[i] != ui_params_expr[i]) {
						is_diff = true; break;
					}
				}
			}
			if (!is_diff) {
				for (int i = 0; i < n_params_pose; i++) {
					if (params_pose[i] != ui_params_pose[i]) {
						is_diff = true; break;
					}
				}
			}
			if (!is_diff) {
				for (int i = 0; i < n_params_eyelid; i++) {
					if (params_eyelid[i] != ui_params_eyelid[i]) {
						is_diff = true; break;
					}
				}
			}
			if (is_diff) Copy();
			return is_diff;
		}
	}

	void FaceParams::Copy() {
		for (int i = 0; i < n_params_shape; i++) {
			ui_params_shape[i] = params_shape[i];
		}
		for (int i = 0; i < n_params_expr; i++) {
			ui_params_expr[i] = params_expr[i];
		}
		for (int i = 0; i < n_params_pose; i++) {
			ui_params_pose[i] = params_pose[i];
		}
		for (int i = 0; i < n_params_eyelid; i++) {
			ui_params_eyelid[i] = params_eyelid[i];
		}
	}

	//////////

	int64_t numel(const std::vector<int64_t>& size) {
		if (size.size()) {
			int64_t n = 1;
			for (size_t i = 0; i < size.size(); i++) {
				n *= size[i];
			}
			return n;
		}
		return 0;
	}

	void get_str(FILE* fp, char* buf) {
		int id = 0;
		do {
			fread(&buf[id++], 1, 1, fp);
		} while (buf[id-1] != 0);
	}

	void parse_record(FILE* f, std::string& name, std::string& type, 
		std::string & content,
		std::vector<int64_t>& shape)
	{
		char cbuff[512];
		get_str(f,cbuff);
		name = cbuff;
		get_str(f,cbuff);
		type = cbuff;
		if (type == "str") {
			get_str(f, cbuff);
			content = cbuff;
		}		
		else if (type == "float32" || type=="float64" || type=="int32" || type=="int64") {
			int64_t dims;
			fread(&dims, sizeof(int64_t), 1, f);
			shape.resize(dims);
			for (int64_t i = 0; i < dims; i++) {
				fread(&shape[i], sizeof(int64_t), 1, f);
			}		
		}
	}

	void TorchFrame::load(const char* filename) {
		FILE* fp;
		fp = fopen(filename, "rb");
		if (fp == 0)
			throw std::runtime_error("load fail!");
#ifdef _WIN32
		_fseeki64(fp, 0, SEEK_END);
		int64_t filesize = _ftelli64(fp);
		int64_t cur_offset = 0;
		_fseeki64(fp, 0, SEEK_SET);
#else
		fseeko64(fp, 0, SEEK_END);
		int64_t filesize = ftello64(fp);
		int64_t cur_offset = 0;
		fseeko64(fp, 0, SEEK_SET);
#endif		
		std::string name, type, content;
		std::vector<int64_t> arr_size;
		while(cur_offset < filesize) {
			parse_record(fp, name, type, content, arr_size);
			//printf("%s %s\n", name.c_str(), type.c_str());
			if (name == "img_size") {
				assert(type == "int64");
				fread(img_size, sizeof(int64_t), 2, fp);
			}
			else if (name == "frame_id") {
				assert(type == "str");
				frame_id = content;
			}
			else if (name == "global_step") {
				assert(type == "int");
				fread(&global_step, sizeof(int64_t), 1, fp);
			}
			else {
				//printf("%s\n", type.c_str());
				assert(type == "float32");
				int64_t read_total = numel(arr_size);
				std::vector<float> loc_data;
				loc_data.resize(read_total);
				fread(loc_data.data(), sizeof(float), read_total, fp);
				params.insert({name, std::move(loc_data)});
			}
#ifdef _WIN32
			cur_offset = _ftelli64(fp);
#else
			cur_offset = ftello64(fp);
#endif
		} 

		fclose(fp);
	}


	///////

	void load(const char* filename, std::vector<FUFrame>& frames)
	{
		const char* whitespace = " \t\n\r";
		const int buff_length = 8192;
		FILE* file_stream;
		char cbuff[buff_length];
		char * current_token = NULL;
		
		file_stream = fopen(filename, "r");
		if (file_stream == 0) {
			sprintf(cbuff, "fail to load %s!\n", filename);
			throw std::runtime_error(cbuff);
		}

		frames.clear();

		int line_number = 0;
		int obj_per_line;
		int ii;

		bool err_flag;
		
		while (fgets(cbuff, buff_length, file_stream)) {
			
			current_token = strtok(cbuff, whitespace);// 
			line_number++;

			//skip comments
			if (current_token == NULL || current_token[0] == '#')
				continue;

			err_flag = false;
			
			while (true) {
				//// read id
				int id = atoi(current_token); obj_per_line = 1;
				if (id >= frames.size()) {
					frames.resize(id + 1);
				}
				FUFrame& frame = frames[id];
				frame.id = id;
				frame.is_valid = false;
				{
					// set default camera
					frame.params_rot[0] = frame.params_rot[1] = frame.params_rot[2] = 0.f;
					frame.params_rot[3] = 1.f;

					frame.params_trans[0] = 0.f;
					frame.params_trans[1] = 0.f;
					frame.params_trans[2] = -1.663f;
				}

				//// read expr
				for (ii = 0; ii < FUFrame::n_params_expr; ii++) {
					current_token = strtok(NULL, whitespace);
					if (current_token == NULL || current_token[0] == '#')
						break;
					obj_per_line++;
					frame.params_expr[ii] = atof(current_token);
				}
				if (ii < FUFrame::n_params_expr) {
					if (ii > 0) // A invalid frame only has index, but we do not throw exception in this case!
						err_flag = true;
					else
						printf("Warning, invalid frame index = %d\n", id);
					break;
				}

				//// read rot
				for (ii = 0; ii < 4; ii++) {
					current_token = strtok(NULL, whitespace);
					if (current_token == NULL || current_token[0] == '#')
						break;
					obj_per_line++;
					frame.params_rot[ii] = atof(current_token);
				}
				if (ii < 4) {
					err_flag = true;  break;
				}

				// read trans
				for (ii = 0; ii < 3; ii++) {
					current_token = strtok(NULL, whitespace);
					if (current_token == NULL || current_token[0] == '#')
						break;
					obj_per_line++;
					frame.params_trans[ii] = atof(current_token);
				}
				if (ii < 3) {
					err_flag = true;  break;
				}

				// read eye_rot
				for (ii = 0; ii < 4; ii++) {
					current_token = strtok(NULL, whitespace);
					if (current_token == NULL || current_token[0] == '#')
						break;
					obj_per_line++;
					frame.params_eye_rot[ii] = atof(current_token);
				}
				if (ii < 4) {
					err_flag = true; break;
				}
				// assign right eye == left eye
				for (int j = 0; j < 4; j++)
					frame.params_eye_rot[4 + j] = frame.params_eye_rot[j];

				// read identity
				for (ii = 0; ii < FUFrame::n_params_shape; ii++) {
					current_token = strtok(NULL, whitespace);
					if (current_token == NULL || current_token[0] == '#')
						break;
					obj_per_line++;
					frame.params_shape[ii] = atof(current_token);
				}
				if (ii < FUFrame::n_params_shape) {
					err_flag = true; break;
				}
				
				current_token = strtok(NULL, whitespace); // Should not read anything
				if (current_token == NULL || current_token[0] == '#') {
					frame.is_valid = true;
					break;
				}

				sprintf(cbuff, "Invalid content! too many items in line %d\n", line_number);
				throw std::runtime_error(cbuff);
				break;
			}

			if (err_flag) {
				sprintf(cbuff, "Invalid content! read %d items in line %d\n", obj_per_line, line_number);
				throw std::runtime_error(cbuff);
			}
		}
		fclose(file_stream);
	}

	void FUFrame::get_camera(float* w2c) const {
		
		// w,x,y,z
		Eigen::Quaternionf quat(
			params_rot[3], params_rot[0], params_rot[1], params_rot[2]
		);
		//Eigen::Quaternionf quat(
			//params_rot[3] * 10,
			//params_rot[0] * 10,
			//params_rot[1] * 10,
			//params_rot[2] * 10
		//);
		//Eigen::Matrix3f mat = quat.toRotationMatrix(); // may not give correct ans when quaternion is normalized
		Eigen::Matrix3f mat = quat.normalized().toRotationMatrix();
	
		w2c[0] = mat(0, 0); w2c[4] = mat(0, 1); w2c[8] = mat(0, 2); w2c[12] = params_trans[0];
		w2c[1] = mat(1, 0); w2c[5] = mat(1, 1); w2c[9] = mat(1, 2); w2c[13] = params_trans[1];
		w2c[2] = mat(2, 0); w2c[6] = mat(2, 1); w2c[10] = mat(2, 2); w2c[14] = params_trans[2];
		w2c[3] = 0.f; w2c[7] = 0.f; w2c[11] = 0.f; w2c[15] = 1.f;
	}

	void FUFrame::get_eyemat(int is_right, float* mat3x3, bool transpose) const {
		// w,x,y,z
		Eigen::Quaternionf quat(
			params_eye_rot[4 * is_right + 3],
			params_eye_rot[4 * is_right + 0],
			params_eye_rot[4 * is_right + 1],
			params_eye_rot[4 * is_right + 2]
		);
		Eigen::Matrix3f mat = quat.normalized().toRotationMatrix();
		if (transpose) {
			mat3x3[0] = mat(0, 0); mat3x3[1] = mat(0, 1); mat3x3[2] = mat(0, 2);
			mat3x3[3] = mat(1, 0); mat3x3[4] = mat(1, 1); mat3x3[5] = mat(1, 2);
			mat3x3[6] = mat(2, 0); mat3x3[7] = mat(2, 1); mat3x3[8] = mat(2, 2);
		}
		else {
			mat3x3[0] = mat(0, 0); mat3x3[3] = mat(0, 1); mat3x3[6] = mat(0, 2);
			mat3x3[1] = mat(1, 0); mat3x3[4] = mat(1, 1); mat3x3[7] = mat(1, 2);
			mat3x3[2] = mat(2, 0); mat3x3[5] = mat(2, 1); mat3x3[8] = mat(2, 2);
		}
	}

	///////

	void CameraVisualizer::update(float length)
	{
		if (v.size() == 0)
			return;
		delete[] vertex_buffer;
		delete[] frame_buffer;
		vertex_buffer = new float[3 * v.size()];
		frame_buffer = new float[6 * v.size() * 3];

		for (size_t i = 0; i < v.size(); i++) {
			float vx = v[i].c2w[9];
			float vy = v[i].c2w[10];
			float vz = v[i].c2w[11];

			//// vertex buffer
			vertex_buffer[3 * i + 0] = vx;
			vertex_buffer[3 * i + 1] = vy;
			vertex_buffer[3 * i + 2] = vz;

			//// frame buffer
			//0,2,4
			frame_buffer[0 + 6 * i + 0] = vx;
			frame_buffer[0 + 6 * i + 1] = vy;
			frame_buffer[0 + 6 * i + 2] = vz;

			frame_buffer[(v.size() + i) * 6 + 0] = vx;
			frame_buffer[(v.size() + i) * 6 + 1] = vy;
			frame_buffer[(v.size() + i) * 6 + 2] = vz;

			frame_buffer[(v.size() * 2 + i) * 6 + 0] = vx;
			frame_buffer[(v.size() * 2 + i) * 6 + 1] = vy;
			frame_buffer[(v.size() * 2 + i) * 6 + 2] = vz;

			// x axis
			frame_buffer[0 + 6 * i + 3] = vx + v[i].c2w[0] * length;
			frame_buffer[0 + 6 * i + 4] = vy + v[i].c2w[1] * length;
			frame_buffer[0 + 6 * i + 5] = vz + v[i].c2w[2] * length;

			frame_buffer[(v.size() + i) * 6 + 3] = vx + v[i].c2w[3] * length;
			frame_buffer[(v.size() + i) * 6 + 4] = vy + v[i].c2w[4] * length;
			frame_buffer[(v.size() + i) * 6 + 5] = vz + v[i].c2w[5] * length;

			frame_buffer[(v.size() * 2 + i) * 6 + 3] = vx + v[i].c2w[6] * length;
			frame_buffer[(v.size() * 2 + i) * 6 + 4] = vy + v[i].c2w[7] * length;
			frame_buffer[(v.size() * 2 + i) * 6 + 5] = vz + v[i].c2w[8] * length;
		}	
	}

	CameraVisualizer::~CameraVisualizer()
	{
		delete[] vertex_buffer;
		delete[] frame_buffer;
	}

	void CameraVisualizer::scale(float sx, float sy, float sz)
	{
		for (size_t i = 0; i < v.size(); i++) {
			SimpleCameraFrame& p = v[i];
			p.c2w[9] *= sx;
			p.c2w[10] *= sy;
			p.c2w[11] *= sz;
		}	
	}

	static void inv_of_unity16(float* in16, float* out16) {
		// [R',-R't;0,1]
		out16[0] = in16[0]; out16[1] = in16[4]; out16[2] = in16[8]; out16[3] = 0.f;
		out16[4] = in16[1]; out16[5] = in16[5]; out16[6] = in16[9]; out16[7] = 0.f;
		out16[8] = in16[2]; out16[9] = in16[6]; out16[10] = in16[10]; out16[11] = 0.f;
		out16[12] = -(in16[12] * out16[0] + in16[13] * out16[4] + in16[14] * out16[8]);
		out16[13] = -(in16[12] * out16[1] + in16[13] * out16[5] + in16[14] * out16[9]);
		out16[14] = -(in16[12] * out16[2] + in16[13] * out16[6] + in16[14] * out16[10]);
		out16[15] = 1.f;
	}

	bool CameraVisualizer::load(const TorchFrame* ptr, int n) {
		
		printf("load camera traj from memory\n");

		this->v.clear();

		for (int i = 0; i < n; i++) {
			
			const TorchFrame* frame = ptr + i;
			{
				const std::vector<float>& R = frame->params.at("R");// ["R"] ;
				const std::vector<float>& t = frame->params.at("t");// ["t"] ;
				float w2c[16], c2w[16];
				w2c[0] = R[0]; w2c[4] = R[1]; w2c[8] = R[2]; w2c[12] = t[0];
				w2c[1] = R[3]; w2c[5] = R[4]; w2c[9] = R[5]; w2c[13] = t[1];
				w2c[2] = R[6]; w2c[6] = R[7]; w2c[10] = R[8]; w2c[14] = t[2];
				w2c[3] = 0; w2c[7] = 0; w2c[11] = 0; w2c[15] = 1;
				inv_of_unity16(w2c, c2w);
				SimpleCameraFrame scf;
				scf.c2w[0] = c2w[0]; scf.c2w[1] = c2w[1]; scf.c2w[2] = c2w[2];
				scf.c2w[3] = c2w[4]; scf.c2w[4] = c2w[5]; scf.c2w[5] = c2w[6];
				scf.c2w[6] = c2w[8]; scf.c2w[7] = c2w[9]; scf.c2w[8] = c2w[10];
				scf.c2w[9] = c2w[12]; scf.c2w[10] = c2w[13]; scf.c2w[11] = c2w[14];
				this->v.push_back(scf);
			}		

			//if (i % 100 == 99) {
			//	printf("[%d/%d]\n", i + 1, n);
			//}
		}

		//printf("done!\n");
		return true;

	}

	bool CameraVisualizer::load(const char* filename) {
		
		std::vector<std::string> filename_list = getFilenameInDirectory(filename, 2); // get only files ?
		if (filename_list.size() == 0) {
			throw std::runtime_error("load vis camera fail!");
		}
		//for (int i = 0; i < filename_list.size(); i++) {
		//	printf("%s\n", filename_list[i].c_str());
		//}

		printf("load camera traj...\n");

		this->v.clear();

		gaussian_splatting::TorchFrame * frame;
		for (int i = 0; i < filename_list.size(); i++) {
			frame = new gaussian_splatting::TorchFrame();
			frame->load(cat_path(filename, filename_list[i].c_str()).c_str());
			{
				std::vector<float>& R = frame->params["R"];
				std::vector<float>& t = frame->params["t"];
				float w2c[16], c2w[16];
				w2c[0] = R[0]; w2c[4] = R[1]; w2c[8] = R[2]; w2c[12] = t[0];
				w2c[1] = R[3]; w2c[5] = R[4]; w2c[9] = R[5]; w2c[13] = t[1];
				w2c[2] = R[6]; w2c[6] = R[7]; w2c[10] = R[8]; w2c[14] = t[2];
				w2c[3] = 0; w2c[7] = 0; w2c[11] = 0; w2c[15] = 1;
				inv_of_unity16(w2c, c2w);
				SimpleCameraFrame scf;
				scf.c2w[0] = c2w[0]; scf.c2w[1] = c2w[1]; scf.c2w[2] = c2w[2];
				scf.c2w[3] = c2w[4]; scf.c2w[4] = c2w[5]; scf.c2w[5] = c2w[6];
				scf.c2w[6] = c2w[8]; scf.c2w[7] = c2w[9]; scf.c2w[8] = c2w[10];
				scf.c2w[9] = c2w[12]; scf.c2w[10] = c2w[13]; scf.c2w[11] = c2w[14];
				this->v.push_back(scf);
			}
			delete frame;
			if (i % 100 == 99) {
				printf("[%d/%d]\n", i + 1, (int)filename_list.size());
			}
		}

		printf("done!\n");

		return true;
	}

	bool CameraVisualizer::load(const std::vector<FUFrame>& frames) {

		printf("load camera traj from FUFrames\n");

		const int n_frames = frames.size();
		this->v.resize(n_frames);

		float w2c[16];
		float c2w[16];

		for (int i = 0; i < n_frames; i++) {
			const FUFrame* frame = frames.data() + i;

			//if (frame->is_valid) {

			frame->get_camera(w2c);
			inv_of_unity16(w2c, c2w);

			//}
			//else {
			//	// create dummy frame
			//	const float R = 1.663f;
			//	c2w[0] = 1.f; c2w[4] = 0.f; c2w[8] = 0.f; c2w[12] = 0.f;
			//	c2w[1] = 0.f; c2w[5] = 1.f; c2w[9] = 0.f; c2w[13] = 0.f;
			//	c2w[2] = 0.f; c2w[6] = 0.f; c2w[10] = 1.f; c2w[14] = R;
			//	c2w[3] = 0.f; c2w[7] = 0.f; c2w[11] = 0.f; c2w[15] = 1.f;
			//}

			SimpleCameraFrame& scf = this->v[i];
			scf.c2w[0] = c2w[0]; scf.c2w[1] = c2w[1]; scf.c2w[2] = c2w[2];
			scf.c2w[3] = c2w[4]; scf.c2w[4] = c2w[5]; scf.c2w[5] = c2w[6];
			scf.c2w[6] = c2w[8]; scf.c2w[7] = c2w[9]; scf.c2w[8] = c2w[10];
			scf.c2w[9] = c2w[12]; scf.c2w[10] = c2w[13]; scf.c2w[11] = c2w[14];
		}
		return true;
	}

	void draw_camera_pos(const CameraVisualizer& p, int render_mode, int offset, int size)
	{
		if (p.v.size() == 0)
			return;
		if (p.vertex_buffer == 0)
			return;

		if (size == -1) {
			size = (int)p.v.size() - offset;
		}
		//int n_size = (int)p.v.size();

		glVertexPointer(3, GL_FLOAT, 0, p.vertex_buffer);
		glEnableClientState(GL_VERTEX_ARRAY);
		if (render_mode == 0) {
			glDrawArrays(GL_POINTS, offset, size);
		}
		else {
			glDrawArrays(GL_LINE_STRIP, offset, size);
		}
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	void draw_camera_frame(const CameraVisualizer& p, int sel) {

		if (p.v.size() == 0)
			return;
		if (p.frame_buffer == 0)
			return;

		glVertexPointer(3, GL_FLOAT, 0, p.frame_buffer);
		glEnableClientState(GL_VERTEX_ARRAY);

		float rgba[4];
		glGetFloatv(GL_CURRENT_COLOR, rgba);
		int n_size = (int)p.v.size();
		if (sel < 0) {
			glColor3f(1.f, 0.f, 0.f);//red
			glDrawArrays(GL_LINES, 0, 2 * n_size);
			glColor3f(0.f, 1.f, 0.f);//green
			glDrawArrays(GL_LINES, 2 * n_size, 2 * n_size);
			glColor3f(0.f, 0.f, 1.f);//blue
			glDrawArrays(GL_LINES, 4 * n_size, 2 * n_size);
		}
		else {
			glColor3f(1.f, 0.f, 0.f);
			glDrawArrays(GL_LINES, sel * 2, 2);
			glColor3f(0.f, 1.f, 0.f);
			glDrawArrays(GL_LINES, (n_size + sel) * 2, 2);
			glColor3f(0.f, 0.f, 1.f);
			glDrawArrays(GL_LINES, (n_size * 2 + sel) * 2, 2);
		}

		glColor3f(rgba[0], rgba[1], rgba[2]);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	//////////////////

	SimpleTexture::SimpleTexture()
	{
		TexID = 0;
		width = 0; height = 0;
		c = 4;
		data = nullptr;
	}

	SimpleTexture::~SimpleTexture() {
		destroyTexture();
		if (data) {
			delete[] data;
			data = nullptr;
		}
	}

	//void SimpleTexture::buildTexture(void * data) {
	//	if (TexID) {
	//		destroyTexture();
	//	}
	//	glGenTextures(1, &TexID);
	//	glBindTexture(GL_TEXTURE_2D, TexID);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST_MIPMAP_LINEAR);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//	//glTexImage2D(GL_TEXTURE_2D, 0, GL)

	//	glBindTexture(GL_TEXTURE_2D, 0);

	//}

	void SimpleTexture::destroyTexture() {
		if (TexID) {
			glDeleteTextures(1, &TexID);
			TexID = 0;
		}
	}

	

}