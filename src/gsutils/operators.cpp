//#include <Windows.h>
#include <fstream>
#include <iostream>
//#include <filesystem>

#include <algorithm>


#include "operators.h"

#include "GL/glew.h"
#include "utils/rot.h"

#ifdef _WIN32
#include "imgui/dirent/dirent.h"
#else
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#endif

#include "utils/Mesh.h"

namespace gaussian_splatting {

	void SetGLTransforms(cuGaussianSplatting::Transforms& trans, bool trans_to_gpu)
	{
		float modelview[16];
		float c2w[16];
		float proj[16];
		float modelproj[16];
		int viewport[4];

		glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)proj);
		glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat*)modelview);
		MatrixTools::mult16(proj, modelview, modelproj);
		MatrixTools::inv_of_unity16(modelview, c2w);
		glGetIntegerv(GL_VIEWPORT, viewport);

		MatrixTools::cp16(modelproj, trans.mvpMatrix);
		MatrixTools::cp16(modelview, trans.mvMatrix);
		MatrixTools::cp16(c2w, trans.c2wMatrix);
		MatrixTools::cp16(proj, trans.projMatrix);
		for (int i = 0; i < 4; i++)
			trans.vpParams[i] = viewport[i];

		if (trans_to_gpu && trans.matrices) {
			trans.CopyData();
		}
	}

	void SetGLTransforms2(cuGaussianSplatting::Transforms& trans, bool trans_to_gpu)
	{
		float modelview[16];
		float c2w[16];
		float proj[16];
		float modelproj[16];
		int viewport[4];

		glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)proj);
		glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat*)modelview);		
		
		// Convert view and projection to target coordinate system
		//auto view_mat = eye.view();
		//auto proj_mat = eye.viewproj();
		//view_mat.row(1) *= -1;
		//view_mat.row(2) *= -1;
		//proj_mat.row(1) *= -1;

		MatrixTools::inv_of_unity16(modelview, c2w);
		MatrixTools::mult16(proj, modelview, modelproj);
		
		modelview[1] *= -1;modelview[5] *= -1; modelview[9] *= -1; modelview[13] *= -1;
		modelview[2] *= -1; modelview[6] *= -1; modelview[10] *= -1; modelview[14] *= -1;
		modelproj[1] *= -1; modelproj[5] *= -1; modelproj[9] *= -1; modelproj[13] *= -1;

		glGetIntegerv(GL_VIEWPORT, viewport);

		MatrixTools::cp16(modelproj, trans.mvpMatrix);
		MatrixTools::cp16(modelview, trans.mvMatrix);
		MatrixTools::cp16(c2w, trans.c2wMatrix);
		MatrixTools::cp16(proj, trans.projMatrix);
		for (int i = 0; i < 4; i++)
			trans.vpParams[i] = viewport[i];

		if (trans_to_gpu && trans.matrices) {
			trans.CopyData();
		}
	}

	void perspective_2_orthogonal(
		const float* align_point,
		const float* w2c,
		const float* proj, float* orthogonal,
		float halfD
	) {
		// compute align point in camera space
		float camera_align_point[3];
		camera_align_point[0] = w2c[0] * align_point[0] + w2c[4] * align_point[1] + w2c[8] * align_point[2] + w2c[12];
		camera_align_point[1] = w2c[1] * align_point[0] + w2c[5] * align_point[1] + w2c[9] * align_point[2] + w2c[13];
		camera_align_point[2] = w2c[2] * align_point[0] + w2c[6] * align_point[1] + w2c[10] * align_point[2] + w2c[14];
		float depth = -camera_align_point[2];
		float kx = proj[0]; // 1./tan(fovy/2)
		float ky = proj[5];
		for (int j = 0; j < 16; j++) {
			orthogonal[j] = 0.f;
		}
		orthogonal[0] = kx / depth;
		orthogonal[5] = ky / depth;
		orthogonal[10] = -1.f / halfD; orthogonal[14] = -depth / halfD;
		orthogonal[15] = 1.f;
	}

	/////////////////////////

	template<typename T>
	T clamp(T v, T v_min, T v_max) {
		if (v < v_min) v = v_min;
		if (v > v_max) v = v_max;
		return v;
	}

	Eigen::Vector3f closesPointOnTriangle(
		const Eigen::Vector3f & x0,
		const Eigen::Vector3f & x1,
		const Eigen::Vector3f & x2,
		const Eigen::Vector3f & sourcePosition,
		float * oS, float * oT
	){
		//vector3 edge0 = triangle[1] - triangle[0];
		//vector3 edge1 = triangle[2] - triangle[0];
		//vector3 v0 = triangle[0] - sourcePosition;
		Eigen::Vector3f edge0 = x1 - x0;
		Eigen::Vector3f edge1 = x2 - x0;
		Eigen::Vector3f v0 = x0 - sourcePosition;

		float a = edge0.dot(edge0);
		float b = edge0.dot(edge1);
		float c = edge1.dot(edge1);
		float d = edge0.dot(v0);
		float e = edge1.dot(v0);

		float det = a * c - b * b;
		float s = b * e - c * d;
		float t = b * d - a * e;

		if (s + t < det)
		{
			if (s < 0.f)
			{
				if (t < 0.f)
				{
					if (d < 0.f)
					{
						s = clamp(-d / a, 0.f, 1.f);
						t = 0.f;
					}
					else
					{
						s = 0.f;
						t = clamp(-e / c, 0.f, 1.f);
					}
				}
				else
				{
					s = 0.f;
					t = clamp(-e / c, 0.f, 1.f);
				}
			}
			else if (t < 0.f)
			{
				s = clamp(-d / a, 0.f, 1.f);
				t = 0.f;
			}
			else
			{
				float invDet = 1.f / det;
				s *= invDet;
				t *= invDet;
			}
		}
		else
		{
			if (s < 0.f)
			{
				float tmp0 = b + d;
				float tmp1 = c + e;
				if (tmp1 > tmp0)
				{
					float numer = tmp1 - tmp0;
					float denom = a - 2 * b + c;
					s = clamp(numer / denom, 0.f, 1.f);
					t = 1 - s;
				}
				else
				{
					t = clamp(-e / c, 0.f, 1.f);
					s = 0.f;
				}
			}
			else if (t < 0.f)
			{
				if (a + d > b + e)
				{
					float numer = c + e - b - d;
					float denom = a - 2 * b + c;
					s = clamp(numer / denom, 0.f, 1.f);
					t = 1 - s;
				}
				else
				{
					s = clamp(-e / c, 0.f, 1.f);
					t = 0.f;
				}
			}
			else
			{
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = clamp(numer / denom, 0.f, 1.f);
				t = 1.f - s;
			}
		}

		*oS = s;
		*oT = t;
		return x0 + s * edge0 + t * edge1;
	}


	void GeneratePointPos(
		const SMesh* mesh,
		const gaussian_splatting::gaussian_points_coord& gcoord,
		local_transfer* loc_transfer,
		gaussian_splatting::gaussian_points& gpc,
		float k
	) {
		const std::vector<Eigen::Vector3f>& vertex_list_src = (loc_transfer->neutral_pos);
		const std::vector<Eigen::Vector3f>& vertex_list_tar = (mesh->vertex_list);
		const std::vector<SMesh::obj_face>& face_list = (mesh->face_list);

		loc_transfer->local_frame.resize(face_list.size());
		loc_transfer->faceR.resize(face_list.size());
		loc_transfer->facet.resize(face_list.size());
		loc_transfer->faceNR.resize(face_list.size());

		const std::vector<Eigen::Vector3f>& params = (loc_transfer->pos_offset);

		///////////////////// process all faces, compute transfer

		for (int i_face = 0; i_face < face_list.size(); i_face++) {
			const SMesh::obj_face& face = face_list[i_face];
			Eigen::Vector3f v0_tar = vertex_list_tar[face.vertex_index[0]];
			Eigen::Vector3f v1_tar = vertex_list_tar[face.vertex_index[1]];
			Eigen::Vector3f v2_tar = vertex_list_tar[face.vertex_index[2]];

			Eigen::Vector3f v0_src = vertex_list_src[face.vertex_index[0]];
			Eigen::Vector3f v1_src = vertex_list_src[face.vertex_index[1]];
			Eigen::Vector3f v2_src = vertex_list_src[face.vertex_index[2]];
		
			//// compute local frame
			Eigen::Vector3f a0_src = v1_src - v0_src;
			Eigen::Vector3f b0 = a0_src.normalized();
			Eigen::Vector3f a1_src = v2_src - v0_src;
			Eigen::Vector3f b1;
			b1 = a1_src - b0.dot(a1_src) * b0;
			b1.normalize();
			Eigen::Vector3f b2 = b0.cross(b1);

			// the data may not need to export
			Eigen::Matrix3f& l_frame = loc_transfer->local_frame[i_face];
			l_frame(0, 0) = b0[0]; l_frame(0, 1) = b0[1]; l_frame(0, 2) = b0[2];
			l_frame(1, 0) = b1[0]; l_frame(1, 1) = b1[1]; l_frame(1, 2) = b1[2];
			l_frame(2, 0) = b2[0]; l_frame(2, 1) = b2[1]; l_frame(2, 2) = b2[2];

			//// compute faceR
			Eigen::Vector3f a2_src = a0_src.cross(a1_src);
			//Eigen::Vector3f pd_src = a2_src / sqrt(a2_src.dot(a2_src)); // TODO, have BUG
			Eigen::Vector3f pd_src = a2_src / sqrt(sqrt(a2_src.dot(a2_src))); // FIX
			Eigen::Vector3f c_src = (v0_src + v1_src + v2_src) / 3.f;
			a2_src = c_src + pd_src - v0_src;

			Eigen::Matrix3f V_src, V_tar;
			V_src(0, 0) = a0_src[0]; V_src(1, 0) = a0_src[1]; V_src(2, 0) = a0_src[2];
			V_src(0, 1) = a1_src[0]; V_src(1, 1) = a1_src[1]; V_src(2, 1) = a1_src[2];
			V_src(0, 2) = a2_src[0]; V_src(1, 2) = a2_src[1]; V_src(2, 2) = a2_src[2];

			Eigen::Vector3f a0_tar = v1_tar - v0_tar;
			Eigen::Vector3f a1_tar = v2_tar - v0_tar;
			Eigen::Vector3f a2_tar = a0_tar.cross(a1_tar);
			//Eigen::Vector3f pd_tar = a2_tar / sqrt(a2_tar.dot(a2_tar)); // TODO, have BUG
			Eigen::Vector3f pd_tar = a2_tar / sqrt(sqrt(a2_tar.dot(a2_tar))); // FIX
			Eigen::Vector3f c_tar = (v0_tar + v1_tar + v2_tar) / 3.f;
			a2_tar = c_tar + pd_tar - v1_src;
			V_tar(0, 0) = a0_tar[0]; V_tar(1, 0) = a0_tar[1]; V_tar(2, 0) = a0_tar[2];
			V_tar(0, 1) = a1_tar[0]; V_tar(1, 1) = a1_tar[1]; V_tar(2, 1) = a1_tar[2];
			V_tar(0, 2) = a2_tar[0]; V_tar(1, 2) = a2_tar[1]; V_tar(2, 2) = a2_tar[2];
			// tar * src-1
			Eigen::Matrix3f& Q = loc_transfer->faceR[i_face];
			Eigen::Vector3f& t = loc_transfer->facet[i_face];
			Q = V_tar * V_src.inverse();
			t = v0_tar - Q * v0_src;
			//// compute faceNR
			Eigen::JacobiSVD<Eigen::Matrix3f> svd;
			//svd.compute(Q, Eigen::ComputeThinV | Eigen::ComputeThinU);
			svd.compute(Q, Eigen::ComputeFullV | Eigen::ComputeFullU);
			// U,S,Vt = svd(Q)
			Eigen::Matrix3f& R = loc_transfer->faceNR[i_face];
			R = svd.matrixU() * svd.matrixV().transpose(); // U * Vt
			if (R.determinant() < 0.f) {
				throw std::runtime_error("Unexpected numerical problem!");
			}
		}

		///////////////////// process all gaussian points

		loc_transfer->pointNR.resize(gcoord.point_coord.size());
		gpc.pos.resize(gcoord.point_coord.size());

		for (int i_point = 0; i_point < gpc.pos.size(); i_point ++) {
			
			const gaussian_splatting::FaceCoord& coord = gcoord.point_coord[i_point];
			int i_face = coord.face_id;
			const SMesh::obj_face& face = face_list[i_face];

			const Eigen::Matrix3f& Q = loc_transfer->faceR[i_face];
			const Eigen::Vector3f& t = loc_transfer->facet[i_face];
			const Eigen::Matrix3f& R = loc_transfer->faceNR[i_face];
			Eigen::Matrix3f& pointR = loc_transfer->pointNR[i_face];
			pointR = R; // just copy

			Eigen::Vector3f v0 = vertex_list_src[face.vertex_index[0]];
			Eigen::Vector3f v1 = vertex_list_src[face.vertex_index[1]];
			Eigen::Vector3f v2 = vertex_list_src[face.vertex_index[2]];

			const Eigen::Matrix3f& l_frame = loc_transfer->local_frame[i_face];
			Eigen::Vector3f t0 = { l_frame(0,0), l_frame(0,1), l_frame(0,2) };
			Eigen::Vector3f t1 = { l_frame(1,0), l_frame(1,1), l_frame(1,2) };
			Eigen::Vector3f t2 = { l_frame(2,0), l_frame(2,1), l_frame(2,2) };

			//// compute final value
			Eigen::Vector3f v = v0 * coord.barycentric_coord[0] + \
				v1 * coord.barycentric_coord[1] + v2 * coord.barycentric_coord[2];
			
			float kx = tanh(params[i_point].x()) * k;
			float ky = tanh(params[i_point].y()) * k;
			float kz = tanh(params[i_point].z()) * k;

			// before transfer
			v = v + t0 * kx + t1 * ky + t2 * kz;
			// after transfer
			v = Q * v + t;
			//if(i < 20)
			//	printf("%.8f %.8f %.8f\n", v[0], v[1], v[2]);
			// write
			gpc.pos[i_point] = v;
		}
	}

	/*
	void GeneratePointPos(
		const SMesh* mesh,
		const gaussian_splatting::gaussian_points_coord& gcoord,
		local_transfer * loc_transfer,
		gaussian_splatting::gaussian_points& gpc,
		float k
	) {
		const std::vector<Eigen::Vector3f>& vertex_list_tar = (mesh->vertex_list);
		const std::vector<SMesh::obj_face>& face_list = (mesh->face_list);
		
		const std::vector<Eigen::Vector3f>& vertex_list_src = (loc_transfer->neutral_pos);
		loc_transfer->local_frame.resize(gcoord.point_coord.size());
		loc_transfer->faceR.resize(gcoord.point_coord.size());
		loc_transfer->facet.resize(gcoord.point_coord.size());
		loc_transfer->faceNR.resize(gcoord.point_coord.size());
		const std::vector<Eigen::Vector3f>& params = (loc_transfer->pos_offset);

		gpc.pos.resize(gcoord.point_coord.size());
		for (int i = 0; i < gpc.pos.size(); i++) {
			const gaussian_splatting::FaceCoord& coord = gcoord.point_coord[i];
			const SMesh::obj_face& face = face_list[coord.face_id];
			Eigen::Vector3f v0_tar = vertex_list_tar[face.vertex_index[0]];
			Eigen::Vector3f v1_tar = vertex_list_tar[face.vertex_index[1]];
			Eigen::Vector3f v2_tar = vertex_list_tar[face.vertex_index[2]];

			Eigen::Vector3f v0_src = vertex_list_src[face.vertex_index[0]];
			Eigen::Vector3f v1_src = vertex_list_src[face.vertex_index[1]];
			Eigen::Vector3f v2_src = vertex_list_src[face.vertex_index[2]];

			//// compute local frame
			Eigen::Vector3f a0_src = v1_src - v0_src;
			Eigen::Vector3f b0 = a0_src.normalized();
			Eigen::Vector3f a1_src = v2_src - v0_src;
			Eigen::Vector3f b1;
			b1 = a1_src - b0.dot(a1_src) * b0;
			b1.normalize();
			Eigen::Vector3f b2 = b0.cross(b1);

			Eigen::Matrix3f& l_frame = loc_transfer->local_frame[i];
			l_frame(0, 0) = b0[0]; l_frame(0, 1) = b0[1]; l_frame(0, 2) = b0[2];
			l_frame(1, 0) = b1[0]; l_frame(1, 1) = b1[1]; l_frame(1, 2) = b1[2];
			l_frame(2, 0) = b2[0]; l_frame(2, 1) = b2[1]; l_frame(2, 2) = b2[2];
			//// compute faceR
			Eigen::Vector3f a2_src = a0_src.cross(a1_src);
			Eigen::Vector3f pd_src = a2_src / sqrt(a2_src.dot(a2_src));
			Eigen::Vector3f c_src = (v0_src + v1_src + v2_src) / 3.f;
			a2_src = c_src + pd_src - v0_src;

			Eigen::Matrix3f V_src,V_tar;
			V_src(0, 0) = a0_src[0]; V_src(1, 0) = a0_src[1]; V_src(2, 0) = a0_src[2];
			V_src(0, 1) = a1_src[0]; V_src(1, 1) = a1_src[1]; V_src(2, 1) = a1_src[2];
			V_src(0, 2) = a2_src[0]; V_src(1, 2) = a2_src[1]; V_src(2, 2) = a2_src[2];

			Eigen::Vector3f a0_tar = v1_tar - v0_tar;
			Eigen::Vector3f a1_tar = v2_tar - v0_tar;
			Eigen::Vector3f a2_tar = a0_tar.cross(a1_tar);
			Eigen::Vector3f pd_tar = a2_tar / sqrt(a2_tar.dot(a2_tar));
			Eigen::Vector3f c_tar = (v0_tar + v1_tar + v2_tar) / 3.f;
			a2_tar = c_tar + pd_tar - v1_src;
			V_tar(0, 0) = a0_tar[0]; V_tar(1, 0) = a0_tar[1]; V_tar(2, 0) = a0_tar[2];
			V_tar(0, 1) = a1_tar[0]; V_tar(1, 1) = a1_tar[1]; V_tar(2, 1) = a1_tar[2];
			V_tar(0, 2) = a2_tar[0]; V_tar(1, 2) = a2_tar[1]; V_tar(2, 2) = a2_tar[2];
			// tar * src-1
			Eigen::Matrix3f& Q = loc_transfer->faceR[i];
			Eigen::Vector3f& t = loc_transfer->facet[i];
			Q = V_tar * V_src.inverse();
			t = v0_tar - Q * v0_src;
			//// compute faceNR
			Eigen::JacobiSVD<Eigen::Matrix3f> svd;
			//svd.compute(Q, Eigen::ComputeThinV | Eigen::ComputeThinU);
			svd.compute(Q, Eigen::ComputeFullV | Eigen::ComputeFullU);
			// U,S,Vt = svd(Q)
			Eigen::Matrix3f& R = loc_transfer->faceNR[i];
			R = svd.matrixU()* svd.matrixV().transpose(); // U * Vt
			if (R.determinant() < 0.f) {
				throw std::runtime_error("Unexpected numerical problem!");
			}
			//// compute final value
			Eigen::Vector3f v = v0_src * coord.barycentric_coord[0] + \
				v1_src * coord.barycentric_coord[1] + v2_src * coord.barycentric_coord[2];
			float kx = tanh(params[i].x()) * k;
			float ky = tanh(params[i].y()) * k;
			float kz = tanh(params[i].z()) * k;

			// before transfer
			v = v + b0 * kx + b1 * ky + b2 * kz;
			// after transfer
			v = Q * v + t; 
			//if(i < 20)
			//	printf("%.8f %.8f %.8f\n", v[0], v[1], v[2]);
			// write
			gpc.pos[i] = v;
		}
	}
	*/

	void GeneratePointPos(
		const SMesh* mesh,
		const gaussian_splatting::gaussian_points_coord& gcoord,
		gaussian_splatting::gaussian_points& gpc
	) {
		const std::vector<Eigen::Vector3f>& vertex_list = (mesh->vertex_list);
		const std::vector<SMesh::obj_face>& face_list = (mesh->face_list);
		gpc.pos.resize(gcoord.point_coord.size());
		for (int i = 0; i < gpc.pos.size(); i++) {
			const gaussian_splatting::FaceCoord& coord = gcoord.point_coord[i];
			const SMesh::obj_face& face = face_list[coord.face_id];
			Eigen::Vector3f v0 = vertex_list[face.vertex_index[0]];
			Eigen::Vector3f v1 = vertex_list[face.vertex_index[1]];
			Eigen::Vector3f v2 = vertex_list[face.vertex_index[2]];
			Eigen::Vector3f v = v0 * coord.barycentric_coord[0] + \
				v1 * coord.barycentric_coord[1] + v2 * coord.barycentric_coord[2];
			gpc.pos[i] = v;
		}
	}

	void get_index(std::vector<int>& index, const char* filename) {
		
		FILE* fp;
		fp = fopen(filename, "r");
		if (fp == 0) {
			//throw std::exception("load in get_index fail!");
			throw std::runtime_error("load in get_index fail!");
		}
		index.clear();
		
		char cbuff[512];
		char* current_token = NULL;
		while (fgets(cbuff, 512, fp)) {
			current_token = strtok(cbuff, " \t\n\r");// 
			//skip comments
			if (current_token == NULL || current_token[0] == '#')
				continue;
			int id = atoi(cbuff);
			index.push_back(id);
		}
		fclose(fp);	
	}

	void rot_match_back_head(
		const std::vector<Eigen::Vector3f>& A,
		const std::vector<Eigen::Vector3f> & B,
		Eigen::Matrix3f & R,
		Eigen::Vector3f & t
	) {
		assert(A.size() == B.size());
		Eigen::Vector3f meanA = { 0,0,0 };
		Eigen::Vector3f meanB = { 0,0,0 };
		for (int i = 0; i < A.size(); i++) {
			meanA = meanA + A[i];
			meanB = meanB + B[i];
		}
		meanA = meanA / (float)A.size();
		meanB = meanB / (float)B.size();
		//std::vector<Eigen::Vector3f> biasA, biasB;
		Eigen::MatrixXf mA(A.size(), 3);
		Eigen::MatrixXf mB(B.size(), 3);
		for (int i = 0; i < A.size(); i++) {
			for (int j = 0; j < 3; j++) {
				mA(i, j) = A[i][j] - meanA[j];
				mB(i, j) = B[i][j] - meanB[j];
			}
		}
		Eigen::MatrixXf H = mA.transpose() * mB;

		//// SVD, return U,S,Vt
		//// R = U * Vt
		////Eigen::MatrixXf m = Eigen::MatrixXf::Random(3, 2);
		//Eigen::MatrixXf m(3,3); 
		//for (int i = 0; i < 9; i++) {
		//	m.data()[i] = i;
		//}
		//std::cout << m << std::endl;
		////Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(m);
		//Eigen::JacobiSVD<Eigen::MatrixXf> svd;
		//svd.compute(m, Eigen::ComputeThinV | Eigen::ComputeThinU);

		//std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
		//std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
		//std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
		
		Eigen::JacobiSVD<Eigen::MatrixXf> svd;
		svd.compute(H, Eigen::ComputeThinV | Eigen::ComputeThinU);
		//R = svd.matrixU() * svd.matrixV().transpose();
		R = svd.matrixV() * svd.matrixU().transpose();
		if (R.determinant() < 0) {
			printf("det(R) < R, reflection detected!, correcting for it ...\n");
			auto b = svd.matrixV();
			b(0, 2) *= -1; // col2 // TODO, need test
			b(1, 2) *= -1;
			b(2, 2) *= -1;
			R = b * svd.matrixU().transpose();
		}
		t = -R * meanA + meanB;	
	}

	Eigen::Matrix3f norm_matrix(const Eigen::Matrix3f & mat) {
		Eigen::JacobiSVD<Eigen::Matrix3f> svd;
		//svd.compute(mat, Eigen::ComputeThinV | Eigen::ComputeThinU);
		svd.compute(mat, Eigen::ComputeFullV | Eigen::ComputeFullU);
		return svd.matrixV() * svd.matrixU().transpose();
	}

	void remove_expr_avg(
		std::vector<Eigen::Vector3f>& out_vertex,
		const FLAME* flame,
		const float* expr_params, int n_expr_params,
		float hack_scale 
	) {
		if (flame->shape_mean == nullptr) {
			printf("Invalid shape mean, no op is done!");
			return;
		}
		const cnpy::NpyArray* v_template = &(flame->v_template);
		const cnpy::NpyArray* shapedirs = &(flame->shapedirs);

		int n_vertex = v_template->shape[0];
		int c = v_template->shape[1];
		int k = shapedirs->shape[2];
		assert(k == 400 && c == 3);

		//printf("I'm called\n");
		for (int i_vertex = 0; i_vertex < n_vertex; i_vertex++) {
			Eigen::Vector3f & vertex = out_vertex[i_vertex];
			double offset[3] = { 0,0,0 };
			for (int ik = 300; ik < 300 + n_expr_params; ik++) {
				float ks = expr_params[ik - 300];
				offset[0] += flame->shape_mean->data<double>()[0 * k + ik] * ks;
				offset[1] += flame->shape_mean->data<double>()[1 * k + ik] * ks;
				offset[2] += flame->shape_mean->data<double>()[2 * k + ik] * ks;
			}
			vertex.x() -= offset[0] * hack_scale;
			vertex.y() -= offset[1] * hack_scale;
			vertex.z() -= offset[2] * hack_scale;
		}	
	}

	double debug_J[5 * 3]; // FIXME: store joint position temporarily
	double debug_outA[5 * 4 * 4]; // FIXME: store rel joint matrix temporarily
	
	void get_jaw_transform(
		const FLAME* flame,
		const float* shape_params, int n_shape_params, // 0-300
		const float* expr_params, int n_expr_params, // 300-400
		const float* pos_params,
		//const float* eyelid_params,
		float hack_scale, float * mat
	) {
		const cnpy::NpyArray* v_template = &(flame->v_template);
		const cnpy::NpyArray* shapedirs = &(flame->shapedirs);
		const cnpy::NpyArray* posedirs = &(flame->posedirs);
		const cnpy::NpyArray* kintree_table = &(flame->kintree_table);
		const cnpy::NpyArray* lbs_weights = &(flame->weights);
		const double* ptr_v_template = v_template->data<double>();
		const double* ptr_shapedirs = shapedirs->data<double>();
		const double* ptr_posedirs = posedirs->data<double>();
		const int* parent = kintree_table->data<int>();
		const double* weights = lbs_weights->data<double>();

		const int n_node = 5;
		int n_vertex = v_template->shape[0];
		int c = v_template->shape[1];
		int k = shapedirs->shape[2];
		assert(k == 400 && c == 3);
		assert(shapedirs->shape[0] == n_vertex && shapedirs->shape[1] == c);

		double* v_shaped = new double[n_vertex * 3];
		double* merged_params = new double[400];
		int i = 0;
		while (i < 300 && i < n_shape_params) {
			merged_params[i] = shape_params[i];
			i++;
		}
		while (i < 300) {
			merged_params[i] = 0;
			i++;
		}

		while (i < 400 && i < n_expr_params + 300) {
			merged_params[i] = expr_params[i - 300];
			i++;
		}
		while (i < 400) {
			merged_params[i] = 0;
			i++;
		}
		//// 1. Add shape contribution
		//	v_shaped = v_template + blend_shapes(betas, shapedirs)
		compute_blendshape<double>(
			v_shaped, merged_params, ptr_v_template, ptr_shapedirs,
			1, n_vertex, k, 3
			);

		//// 2. Get the joints NxJx3
		// J can be affected by identity, but hardly affected by expr.
		double J[n_node * 3]; // [1,5,3]
		flame->J_regressor_.apply(J, v_shaped, 1, n_node, n_vertex, 3);

		//// 3. Add pose blend shape		
		double d_pose[n_node * 3];
		for (int i = 0; i < n_node * 3; i++) {
			d_pose[i] = pos_params[i];
		}
		double rot_mats[n_node * 3 * 3];
		double pose_feature[(n_node - 1) * 3 * 3];
		exp_so3<double>(
			rot_mats, d_pose, n_node
			);
		for (int i = 1; i < n_node; i++) {
			for (int j = 0; j < 3 * 3; j++) { // Copy [1:]
				pose_feature[(i - 1) * 9 + j] = rot_mats[i * 9 + j];
			}
			pose_feature[(i - 1) * 9 + 0] -= 1; // -Iden
			pose_feature[(i - 1) * 9 + 4] -= 1;
			pose_feature[(i - 1) * 9 + 8] -= 1;
		}
		// [5,3,3] -- X[1:] --> [4,3,3] -> [36]
		{
			double* pose_offsets = new double[n_vertex * 3];
			compute_blendshape<double>(
				pose_offsets, pose_feature, nullptr, ptr_posedirs,
				1, n_vertex, (n_node - 1) * 3 * 3, 3
				);
			// v_posed = pose_offsets + v_shaped
			for (int i = 0; i < n_vertex; i++) {
				v_shaped[i * 3 + 0] += pose_offsets[i * 3 + 0];
				v_shaped[i * 3 + 1] += pose_offsets[i * 3 + 1];
				v_shaped[i * 3 + 2] += pose_offsets[i * 3 + 2];
			}
			delete[] pose_offsets;
		}

		//// 4. Get the global joint location
		double rel_joints[n_node * 3]; // compute relative offset to parent node

		for (int i = 0; i < n_node; i++) {
			if (i == 0) {
				for (int j = 0; j < 3; j++)
					rel_joints[i * 3 + j] = J[i * 3 + j]; // simple copy
			}
			else {
				for (int j = 0; j < 3; j++)
					rel_joints[i * 3 + j] = J[i * 3 + j] - J[parent[i] * 3 + j];
			}
		}

		// transforms_mat
		// Warining! the inplace op here may lead to error, the former node should be computed first
		double transforms[n_node * 4 * 4];
		merge_Rt<double>(transforms, rot_mats, rel_joints, n_node);
		for (int i = 1; i < n_node; i++) {
			double loc_buf[16];
			double* in_out = &transforms[i * 4 * 4];
			double* in_parent = &transforms[parent[i] * 4 * 4];
			matrix_mult4x4<double>(loc_buf, in_parent, in_out);
			for (int j = 0; j < 16; j++)
				in_out[j] = loc_buf[j];
		}// loc 2 world transforms ??

		//for (int i = 0; i < 16; i++) {
		//	mat[i] = transforms[i] * 1.30787707e-02 + \
		//		transforms[16 + i] * 4.26467828e-02 + \
		//		transforms[32 + i] * 9.44269622e-01;
		//}

		// rel_transforms = transforms{pos} - transforms @ Joint_in_global_coord
		//double posed_joints[n_node * 3]; // pos of transforms, J_transformed
		double rel_transforms[n_node * 4 * 4]; // A
		//After coordinate transformation, what obtained is the relative position to the center of the Joint in world coordinates.
		for (int i = 0; i < n_node; i++) {
			const double* loc_mat = &transforms[i * 16];
			// copy from
			for (int j = 0; j < 16; j++) {
				rel_transforms[i * 16 + j] = loc_mat[j];
			}
			double loc_p[3];
			mat4x4_applyrot<double>(loc_mat, &J[3 * i], loc_p);
			rel_transforms[i * 16 + 3] -= loc_p[0];
			rel_transforms[i * 16 + 7] -= loc_p[1];
			rel_transforms[i * 16 + 11] -= loc_p[2];
		}

		for (int i = 0; i < 16; i++) {
			mat[i] = rel_transforms[i] * 1.30787707e-02 + \
				rel_transforms[16 + i] * 4.26467828e-02 + \
				rel_transforms[32 + i] * 9.44269622e-01;
		}
	}

	void debug_get_jaw_transform(float * mat4x4) {
		for (int i = 0; i < 16; i++) {
			mat4x4[i] = debug_outA[i] * 1.30787707e-02 + \
				debug_outA[16 + i] * 4.26467828e-02 + \
				debug_outA[32 + i] * 9.44269622e-01;
		}		
	}

	void lbs_acc(
		std::vector<Eigen::Vector3f>& out_vertex,
		const std::vector<int>& sel_id,
		const FLAME* flame,
		const float* shape_params, int n_shape_params, // 0-300
		const float* expr_params, int n_expr_params, // 300-400
		const float* pos_params,
		const float* eyelid_params,
		float hack_scale
	) {
		const cnpy::NpyArray* v_template = &(flame->v_template);
		const cnpy::NpyArray* shapedirs = &(flame->shapedirs);
		const cnpy::NpyArray* posedirs = &(flame->posedirs);
		const cnpy::NpyArray* kintree_table = &(flame->kintree_table);
		const cnpy::NpyArray* lbs_weights = &(flame->weights);
		const double* ptr_v_template = v_template->data<double>();
		const double* ptr_shapedirs = shapedirs->data<double>();
		const double* ptr_posedirs = posedirs->data<double>();
		const int* parent = kintree_table->data<int>();
		const double* weights = lbs_weights->data<double>();

		const int n_node = 5;
		int n_vertex = v_template->shape[0];
		int c = v_template->shape[1];
		int k = shapedirs->shape[2];
		assert(k == 400 && c == 3);
		assert(shapedirs->shape[0] == n_vertex && shapedirs->shape[1] == c);

		//double* v_shaped = new double[n_vertex * 3];

		double* merged_params = new double[400];
		int i = 0;
		while (i < 300 && i < n_shape_params) {
			merged_params[i] = shape_params[i];
			i++;
		}
		while (i < 300) {
			merged_params[i] = 0;
			i++;
		}

		while (i < 400 && i < n_expr_params + 300) {
			merged_params[i] = expr_params[i - 300];
			i++;
		}
		while (i < 400) {
			merged_params[i] = 0;
			i++;
		}

		int  n_keep = sel_id.size();
		std::vector<int> sel_index = sel_id;

		// Add vertex that can affect J node !
		for (int i = 0; i < n_node; i++) {
			auto tmp = flame->J_regressor_.ptr[i];
			int n_items = tmp.n_items;
			for (int j = 0; j < n_items; j++) {
				sel_index.push_back(tmp.items[j].id);
			}
		}

		double* v_shaped_sel = new double[sel_index.size() * 3];

		// 3522,747,1298,745,1894

		// 1.partially compute blendshape
		compute_blendshape_and_index<double>(
			v_shaped_sel, merged_params, ptr_v_template, ptr_shapedirs, sel_index.data(),
			1, n_vertex, k, (int)sel_index.size() ,3
		);

		//printf("acc\n");
		//for (int i = 0; i < 5; i++) {
		//	printf("%f  %f  %f\n", 
		//		v_shaped_sel[i * 3 + 0], 
		//		v_shaped_sel[i * 3 + 1], 
		//		v_shaped_sel[i * 3 + 2]
		//	);
		//}
		
		// 2.partially compute J
		double J[n_node * 3]; // [1,5,3]
		//flame->J_regressor_.apply(J, v_shaped, 1, n_node, n_vertex, 3);
		flame->J_regressor_.op1(
			J, v_shaped_sel + 3 * n_keep, sel_index.data() + n_keep,
			(int)sel_index.size() - n_keep,  n_node, 3
		);

		//printf("acc\n");
		//for (int i = 0; i < 5; i++) {
		//	printf("%f  %f  %f\n", J[i * 3 + 0], J[i * 3 + 1], J[i * 3 + 2]);
		//}
		 
		// 3.add pose blendshape
		double d_pose[n_node * 3];
		for (int i = 0; i < n_node * 3; i++) {
			d_pose[i] = pos_params[i];
		}
		double rot_mats[n_node * 3 * 3];
		double pose_feature[(n_node - 1) * 3 * 3];
		exp_so3<double>(
			rot_mats, d_pose, n_node
			);
		for (int i = 1; i < n_node; i++) {
			for (int j = 0; j < 3 * 3; j++) { // Copy 1:
				pose_feature[(i - 1) * 9 + j] = rot_mats[i * 9 + j];
			}
			pose_feature[(i - 1) * 9 + 0] -= 1; // -Iden
			pose_feature[(i - 1) * 9 + 4] -= 1;
			pose_feature[(i - 1) * 9 + 8] -= 1;
		}
		// [5,3,3] -- X[1:] --> [4,3,3] -> [36]
		{
			//double* pose_offsets = new double[n_vertex * 3];
			double* pose_offsets = new double[n_keep * 3];

			compute_blendshape_and_index<double>(
				pose_offsets, pose_feature, nullptr, ptr_posedirs, sel_index.data(),
				1, n_vertex, (n_node - 1) * 3 * 3, n_keep, 3
			);
			//compute_blendshape<double>(
			//	pose_offsets, pose_feature, nullptr, ptr_posedirs,
			//	1, n_vertex, (n_node - 1) * 3 * 3, 3
			//);
			
			// v_posed = pose_offsets + v_shaped
			for (int i = 0; i < n_keep; i++) {
				v_shaped_sel[i * 3 + 0] += pose_offsets[i * 3 + 0];
				v_shaped_sel[i * 3 + 1] += pose_offsets[i * 3 + 1];
				v_shaped_sel[i * 3 + 2] += pose_offsets[i * 3 + 2];
			}
			//for (int i = 0; i < n_vertex; i++) {
			//	v_shaped[i * 3 + 0] += pose_offsets[i * 3 + 0];
			//	v_shaped[i * 3 + 1] += pose_offsets[i * 3 + 1];
			//	v_shaped[i * 3 + 2] += pose_offsets[i * 3 + 2];
			//}
			delete[] pose_offsets;
		}
		
		// 4.get the global joint location
		double rel_joints[n_node * 3]; // compute relative offset to parent node
		for (int i = 0; i < n_node; i++) {
			if (i == 0) {
				for (int j = 0; j < 3; j++)
					rel_joints[i * 3 + j] = J[i * 3 + j]; // simple copy
			}
			else {
				for (int j = 0; j < 3; j++)
					rel_joints[i * 3 + j] = J[i * 3 + j] - J[parent[i] * 3 + j];
			}
		}
		// transforms_mat
		// Warining! the inplace op here may lead to error, the former node should be computed first
		double transforms[n_node * 4 * 4];
		merge_Rt<double>(transforms, rot_mats, rel_joints, n_node);
		for (int i = 1; i < n_node; i++) {
			double loc_buf[16];
			double* in_out = &transforms[i * 4 * 4];
			double* in_parent = &transforms[parent[i] * 4 * 4];
			matrix_mult4x4<double>(loc_buf, in_parent, in_out);
			for (int j = 0; j < 16; j++)
				in_out[j] = loc_buf[j];
		}// loc 2 world transforms ??
		
		if (true) {
			// DEBUG dump
			for (int i = 0; i < n_node; i++) {
				debug_J[3 * i + 0] = transforms[i * 16 + 3];
				debug_J[3 * i + 1] = transforms[i * 16 + 7];
				debug_J[3 * i + 2] = transforms[i * 16 + 11];
			}
		}

		double rel_transforms[n_node * 4 * 4]; // A
		//After coordinate transformation, what obtained is the relative position to the center of the Joint in world coordinates.
		for (int i = 0; i < n_node; i++) {
			const double* loc_mat = &transforms[i * 16];
			// copy from
			for (int j = 0; j < 16; j++) {
				rel_transforms[i * 16 + j] = loc_mat[j];
			}
			double loc_p[3];
			mat4x4_applyrot<double>(loc_mat, &J[3 * i], loc_p);
			rel_transforms[i * 16 + 3] -= loc_p[0];
			rel_transforms[i * 16 + 7] -= loc_p[1];
			rel_transforms[i * 16 + 11] -= loc_p[2];
		}

		if (true) {
			// Debug dump outA
			for (int i = 0; i < n_node * 4 * 4; i++) {
				debug_outA[i] = rel_transforms[i];
			}
		}

		//// 5. do skinning
		double* T_ = new double[n_keep * 4 * 4];
		for (int i = 0; i < n_keep; i++) {
			int src_i = sel_index[i];
			for (int j = 0; j < 16; j++) {
				double ans = 0;
				for (int i_node = 0; i_node < n_node; i_node++) {
					//ans += weights[i * n_node + i_node] * rel_transforms[i_node * 16 + j];
					ans += weights[src_i * n_node + i_node] * rel_transforms[i_node * 16 + j];
				}
				T_[i * 16 + j] = ans;
			}
		}
		double* out_buffer = new double[n_keep * 3];
		for (int i = 0; i < n_keep; i++) {
			double* outp = &out_buffer[3 * i];
			double vx = v_shaped_sel[3 * i + 0];
			double vy = v_shaped_sel[3 * i + 1];
			double vz = v_shaped_sel[3 * i + 2];
			const double* mat = &T_[16 * i];
			outp[0] = mat[0] * vx + mat[1] * vy + mat[2] * vz + mat[3];
			outp[1] = mat[4] * vx + mat[5] * vy + mat[6] * vz + mat[7];
			outp[2] = mat[8] * vx + mat[9] * vy + mat[10] * vz + mat[11];
		}
		if (eyelid_params) {
			if (flame->l_eyelid) {
				const double* ptr_l_eyelid = flame->l_eyelid->data<double>();
				for (int i = 0; i < n_keep; i++) {
					int src_i = sel_index[i];
					out_buffer[3 * i + 0] += eyelid_params[0] * ptr_l_eyelid[3 * src_i + 0];
					out_buffer[3 * i + 1] += eyelid_params[0] * ptr_l_eyelid[3 * src_i + 1];
					out_buffer[3 * i + 2] += eyelid_params[0] * ptr_l_eyelid[3 * src_i + 2];
				}
			}
			if (flame->r_eyelid) {
				const double* ptr_r_eyelid = flame->r_eyelid->data<double>();
				for (int i = 0; i < n_keep; i++) {
					int src_i = sel_index[i];
					out_buffer[3 * i + 0] += eyelid_params[1] * ptr_r_eyelid[3 * src_i + 0];
					out_buffer[3 * i + 1] += eyelid_params[1] * ptr_r_eyelid[3 * src_i + 1];
					out_buffer[3 * i + 2] += eyelid_params[1] * ptr_r_eyelid[3 * src_i + 2];
				}
			}
		}
		for (int i = 0; i < n_keep; i++) {
			Eigen::Vector3f& outp = out_vertex[i];
			outp[0] = out_buffer[3 * i + 0] * hack_scale;
			outp[1] = out_buffer[3 * i + 1] * hack_scale;
			outp[2] = out_buffer[3 * i + 2] * hack_scale;
		}

		delete[] out_buffer;
		delete[] T_;
		delete[] v_shaped_sel;
		delete[] merged_params;
	}

	void lbs(
		std::vector<Eigen::Vector3f>& out_vertex,
		const FLAME* flame,
		const float* shape_params, int n_shape_params, // 0-300
		const float* expr_params, int n_expr_params, // 300-400
		const float* pos_params,
		const float* eyelid_params,
		float hack_scale
	) {
		const cnpy::NpyArray* v_template = &(flame->v_template);
		const cnpy::NpyArray* shapedirs = &(flame->shapedirs);
		const cnpy::NpyArray* posedirs = &(flame->posedirs);
		const cnpy::NpyArray* kintree_table = &(flame->kintree_table);
		const cnpy::NpyArray* lbs_weights = &(flame->weights);
		const double* ptr_v_template = v_template->data<double>();
		const double* ptr_shapedirs = shapedirs->data<double>();
		const double* ptr_posedirs = posedirs->data<double>();
		const int* parent = kintree_table->data<int>();
		const double* weights = lbs_weights->data<double>();

		const int n_node = 5;
		int n_vertex = v_template->shape[0];
		int c = v_template->shape[1];
		int k = shapedirs->shape[2];
		assert(k == 400 && c == 3);
		assert(shapedirs->shape[0] == n_vertex && shapedirs->shape[1] == c);
				
		double* v_shaped = new double[n_vertex * 3];
		double * merged_params = new double[400];
		int i = 0;
		while (i < 300 && i < n_shape_params) {
			merged_params[i] = shape_params[i];
			i++;
		}
		while (i < 300) {
			merged_params[i] = 0;
			i++;
		}

		while (i < 400 && i < n_expr_params + 300) {
			merged_params[i] = expr_params[i - 300];
			i++;
		}
		while (i < 400) {
			merged_params[i] = 0;
			i++;
		}
		//// 1. Add shape contribution
		//	v_shaped = v_template + blend_shapes(betas, shapedirs)
		compute_blendshape<double>(
			v_shaped, merged_params, ptr_v_template, ptr_shapedirs,
			1, n_vertex, k, 3
		);

		//const int debug_index[] = { 3522,747,1298,745,1894 };
		//printf("ref\n");
		//for (int i = 0; i < 5; i++) {
		//	int j = debug_index[i];
		//	printf("%f  %f  %f\n",
		//		v_shaped[j * 3 + 0],
		//		v_shaped[j * 3 + 1],
		//		v_shaped[j * 3 + 2]
		//	);
		//}

		//// 2. Get the joints NxJx3
		// J can be affected by identity, but hardly affected by expr. 
		double J[n_node * 3]; // [1,5,3]
		flame->J_regressor_.apply(J, v_shaped, 1, n_node, n_vertex, 3); 
		
		//printf("ref\n");
		//for (int i = 0; i < 5; i++) {
		//	printf("%f  %f  %f\n", J[i * 3 + 0], J[i * 3 + 1], J[i * 3 + 2]);
		//}

		//// 3. Add pose blend shape		
		double d_pose[n_node * 3];
		for (int i = 0; i < n_node * 3; i++) {
			d_pose[i] = pos_params[i];
		}		
		double rot_mats[n_node * 3 * 3];
		double pose_feature[(n_node-1) * 3 * 3];
		exp_so3<double>(
			rot_mats, d_pose, n_node
		);		
		for (int i = 1; i < n_node; i++) {
			for (int j = 0; j < 3 * 3; j++) { // Copy 1:
				pose_feature[(i - 1) * 9 + j] = rot_mats[i * 9 + j];
			}
			pose_feature[(i-1) * 9 + 0] -= 1; // -Iden
			pose_feature[(i-1) * 9 + 4] -= 1;
			pose_feature[(i-1) * 9 + 8] -= 1;
		}
		// [5,3,3] -- X[1:] --> [4,3,3] -> [36]
		{
			double* pose_offsets = new double[n_vertex * 3];
			compute_blendshape<double>(
				pose_offsets, pose_feature, nullptr, ptr_posedirs,
				1, n_vertex, (n_node - 1) * 3 * 3, 3
			);
			// v_posed = pose_offsets + v_shaped
			for (int i = 0; i < n_vertex; i++) {
				v_shaped[i * 3 + 0] += pose_offsets[i * 3 + 0];
				v_shaped[i * 3 + 1] += pose_offsets[i * 3 + 1];
				v_shaped[i * 3 + 2] += pose_offsets[i * 3 + 2];
			}
			delete[] pose_offsets;
		}

		//// 4. Get the global joint location
		double rel_joints[n_node * 3]; // compute relative offset to parent node

		for (int i = 0; i < n_node; i++) {
			if (i == 0) {
				for(int j = 0; j < 3; j++)
					rel_joints[i * 3 + j] = J[i * 3 + j]; // simple copy
			}
			else {
				for(int j = 0; j < 3; j++)
					rel_joints[i * 3 + j] = J[i * 3 + j] - J[parent[i] * 3 + j];
			}
		}

		// transforms_mat
		// Warining! the inplace op here may lead to error, the former node should be computed first
		double transforms[n_node * 4 * 4];
		merge_Rt<double>(transforms, rot_mats, rel_joints, n_node);
		for (int i = 1; i < n_node; i++) {
			double loc_buf[16];
			double* in_out = &transforms[i * 4 * 4];
			double* in_parent = &transforms[parent[i] * 4 * 4];
			matrix_mult4x4<double>(loc_buf, in_parent, in_out);
			for (int j = 0; j < 16; j++)
				in_out[j] = loc_buf[j];
		}// loc 2 world transforms ??
		
		if (true) {
			// DEBUG dump
			for (int i = 0; i < n_node; i++) {
				debug_J[3 * i + 0] = transforms[i * 16 + 3];
				debug_J[3 * i + 1] = transforms[i * 16 + 7];
				debug_J[3 * i + 2] = transforms[i * 16 + 11];
			}
		}

		// rel_transforms = transforms{pos} - transforms @ Joint_in_global_coord
		//double posed_joints[n_node * 3]; // pos of transforms, J_transformed
		double rel_transforms[n_node * 4 * 4]; // A
		//After coordinate transformation, what obtained is the relative position to the center of the Joint in world coordinates.
		for (int i = 0; i < n_node; i++) {
			const double* loc_mat = &transforms[i * 16];
			// copy from
			for (int j = 0; j < 16; j++) {
				rel_transforms[i * 16 + j] = loc_mat[j];
			}
			double loc_p[3];
			mat4x4_applyrot<double>(loc_mat, &J[3 * i], loc_p); 
			rel_transforms[i * 16 + 3] -= loc_p[0];
			rel_transforms[i * 16 + 7] -= loc_p[1];
			rel_transforms[i * 16 + 11] -= loc_p[2];
		}

		if (true) { 
			// Debug dump outA
			for (int i = 0; i < n_node * 4 * 4; i++) {
				debug_outA[i] = rel_transforms[i];
			}		
		}

		//// 5. Do skinning
		double * T_ = new double[n_vertex * 4 * 4];
		for (int i = 0; i < n_vertex; i++) { 
			for (int j = 0; j < 16; j++) {
				double ans = 0;
				for (int i_node = 0; i_node < n_node; i_node++) {
					ans += weights[i * n_node + i_node] * rel_transforms[i_node * 16 + j];
				}
				T_[i * 16 + j] = ans;
			}
		}
		double * out_buffer = new double[n_vertex * 3];
		for (int i = 0; i < n_vertex; i++) {
			//Eigen::Vector3f& outp = out_vertex[i];
			double* outp = &out_buffer[3 * i];
			double vx = v_shaped[3 * i + 0];
			double vy = v_shaped[3 * i + 1];
			double vz = v_shaped[3 * i + 2];
			const double* mat = &T_[16 * i];
			outp[0] = mat[0] * vx + mat[1] * vy + mat[2] * vz + mat[3];
			outp[1] = mat[4] * vx + mat[5] * vy + mat[6] * vz + mat[7];
			outp[2] = mat[8] * vx + mat[9] * vy + mat[10] * vz + mat[11];
		}
		if (eyelid_params) {
			if (flame->l_eyelid) {
				const double* ptr_l_eyelid = flame->l_eyelid->data<double>();
				for (int i = 0; i < n_vertex; i++) {
					out_buffer[3 * i + 0] += eyelid_params[0] * ptr_l_eyelid[3 * i + 0];
					out_buffer[3 * i + 1] += eyelid_params[0] * ptr_l_eyelid[3 * i + 1];
					out_buffer[3 * i + 2] += eyelid_params[0] * ptr_l_eyelid[3 * i + 2];
				}
			}
			if (flame->r_eyelid) {
				const double* ptr_r_eyelid = flame->r_eyelid->data<double>();
				for (int i = 0; i < n_vertex; i++) {
					out_buffer[3 * i + 0] += eyelid_params[1] * ptr_r_eyelid[3 * i + 0];
					out_buffer[3 * i + 1] += eyelid_params[1] * ptr_r_eyelid[3 * i + 1];
					out_buffer[3 * i + 2] += eyelid_params[1] * ptr_r_eyelid[3 * i + 2];
				}
			}
		}
		for (int i = 0; i < n_vertex; i++) {
			Eigen::Vector3f& outp = out_vertex[i];
			outp[0] = out_buffer[3 * i + 0] * hack_scale;
			outp[1] = out_buffer[3 * i + 1] * hack_scale;
			outp[2] = out_buffer[3 * i + 2] * hack_scale;
		}
		//if (true) {
		//	cnpy::NpyArray outnpy = cnpy::NpyArray({(uint64_t)n_vertex,3}, sizeof(float), false);
		//	float* ptr_outnpy = outnpy.data<float>();
		//	for (int i = 0; i < n_vertex; i++) {
		//		ptr_outnpy[i * 3 + 0] = out_vertex[i][0];
		//		ptr_outnpy[i * 3 + 1] = out_vertex[i][1];
		//		ptr_outnpy[i * 3 + 2] = out_vertex[i][2];
		//	}
		//	cnpy::npy_save("D:\\PycharmProjects\\face_code\\photometric_optimization\\result2_C.npy", (float *)&out_vertex[0], { (uint64_t)n_vertex,3 });
		//}
		delete[] out_buffer;
		delete[] T_;
		
		delete[] v_shaped;
		delete[] merged_params;		
	}


	/////////////////////////

	bool is_root_path(const char* filename) {
		if (strlen(filename) < 2)
			return false;
		if ((filename[0] >= 'a' && filename[0] <= 'z') ||
			(filename[0] >= 'A' && filename[0] <= 'Z'))
		{
			if (filename[1] == ':')
				return true;
		}
		return false;
	}

	std::string cat_path(const char* path, const char* file)
	{
		size_t len = strlen(path);
		size_t len2 = strlen(file);
		bool add_flag = true;
		if (len > 0) {
			if (path[len - 1] == '\\' || path[len - 1] == '/')
				add_flag = false;
		}
		else
			add_flag = false;
		std::string out;
		out.resize(len + len2 + (add_flag ? 1 : 0));
		size_t idx = 0;
		while (idx < len) {
			out[idx] = path[idx];
			idx++;
		}
		if (add_flag)
			out[idx++] = '/';
		size_t idx_ = 0;
		while (idx_ < len2) {
			out[idx++] = file[idx_++];
		}
		return out;
	}

	std::string to_lower(const std::string& s) {
		std::string out = s;
		size_t N = out.length();
		for (size_t i = 0; i < N; i++) {
			if (out[i] >= 'A' && out[i] <= 'Z') {
				out[i] = (unsigned char)(int(out[i]) - 65 + 97);
			}
		}
		return out;
	}

	static int find_lastSlash(const char* filename) {
		int ans = -1;
		const char* f = filename;
		int pos = 0;
		while (*f) {
			if (*f == '\\' || *f == '/')
				ans = pos;
			f++; pos++;
		}
		return ans;
	}

	void path_split(const char* filename, char* path, char* file) {
		int slash_pos = find_lastSlash(filename);
		// copy path
		int idx = 0;
		for (; idx < slash_pos; idx++) {
			path[idx] = filename[idx];
		}
		path[idx] = 0;
		// copy file
		idx = slash_pos + 1;
		int i = 0;
		while (filename[idx]) {
			file[i++] = filename[idx++];
		}
		file[i] = 0;
	}

	bool isValidFile(const char* filename)
	{
		std::ifstream infile(filename);
		//infile.open(filename);
		bool flag = infile.good();
		return flag;
	}

	// void createDirectoryRecursively(const std::string& directory) {
		// // Attempt to create the directory and all necessary parent directories
		// if (std::filesystem::create_directories(directory)) {
			// //std::cout << "Directory created: " << directory << std::endl;
			// return;
		// }
		// else {
			// //std::cout << "Directory already exists or cannot be created: " << directory << std::endl;
			// throw std::runtime_error("Directory already exists or cannot be created");
			// //return false;
		// }
	// }
	
	// std::vector<std::string> getFilenameInDirectory(const char * dirname, int flag) {

		// std::vector<std::string> output;

		// // Check if the directory exists and is a directory
		// if (std::filesystem::exists(dirname) && std::filesystem::is_directory(dirname)) {
			// // Iterate over the directory entries
			// for (const auto& entry : std::filesystem::directory_iterator(dirname)) {
				// if (flag == 1) { // keep directory
					// if (!entry.is_regular_file())
						// //output.push_back(entry.path());
						// output.push_back(std::filesystem::relative(entry.path(),dirname));
				// }
				// else if (flag == 2) {
					// if (entry.is_regular_file())
						// //output.push_back(entry.path());
						// output.push_back(std::filesystem::relative(entry.path(),dirname));
				// }
				// else {
					// //output.push_back(entry.path());
					// output.push_back(std::filesystem::relative(entry.path(),dirname));
				// }
			// }
		// }
		// else {
			// throw std::runtime_error(
				// "Provided path is not a directory or does not exists."
			// );
			// //std::cerr << "Provided path is not a directory or does not exist." << std::endl;
		// }
		// std::sort(output.begin(),output.end());
		// return output;
	// }
	
#ifdef _WIN32
	// Windows
	
	void createDirectoryRecursively(const std::string& directory) {
		//static const std::wstring separators(L"\\/");
		static const std::string separators("\\/");

		// If the specified directory name doesn't exist, do our thing
		DWORD fileAttributes = ::GetFileAttributesA(directory.c_str());
		if (fileAttributes == INVALID_FILE_ATTRIBUTES) {

			// Recursively do it all again for the parent directory, if any
			std::size_t  slashIndex = directory.find_last_of(separators);
			if (slashIndex != std::wstring::npos) {
				createDirectoryRecursively(directory.substr(0, slashIndex));
			}

			// Create the last directory on the path (the recursive calls will have taken
			// care of the parent directories by now)
			BOOL result = ::CreateDirectoryA(directory.c_str(), nullptr);
			if (result == FALSE) {
				throw std::runtime_error("Could not create directory");
			}

		}
		else { // Specified directory name already exists as a file or directory

			bool isDirectoryOrJunction =
				((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) ||
				((fileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0);

			if (!isDirectoryOrJunction) {
				throw std::runtime_error(
					"Could not create directory because a file with the same name exists"
				);
			}
		}
	}

	std::vector<std::string> getFilenameInDirectory(const char* dirname, int flag) {

		std::vector<std::string> output;

		DIR* dir;
		struct dirent* ent;
		struct stat s;
		char buffer[256];

		if ((dir = opendir(dirname)) != NULL) {
			/* print all the files and directories within directory */
			while ((ent = readdir(dir)) != NULL) {

				//printf("%s\n", ent->d_name);
				std::string _locn = ent->d_name;

				// remove . && .. 
				if (_locn.length() == 1 && _locn[0] == '.')
					continue;
				else if (_locn.length() == 2 && _locn[0] == '.' && _locn[1] == '.')
					continue;
				else {
					std::string full_name = cat_path(dirname, _locn.c_str());
					if (stat(full_name.c_str(), &s) == 0) {
						if (flag == 1) { // keep directory
							if (!(s.st_mode & S_IFDIR)) { // not directory
								continue;
							}
						}
						else if (flag == 2) { // keep file
							if (!(s.st_mode & S_IFREG)) { // not file
								continue;
							}
						}
					}
					else {
						sprintf_s(buffer, "Unknown error when stat file %s", full_name.c_str());
						throw std::runtime_error(buffer);
					}
				}
				output.push_back(_locn);
			}
			closedir(dir);
		}
		else {
			/* could not open directory */
			throw std::runtime_error(
				"Could not read directory"
			);
		}
		return std::move(output);
	}
#else
	// Linux
	void createDirectoryRecursively(const std::string& directory) {
		size_t prev = 0;
		size_t pos;
		std::string dir;
		int mdret;

		// Check if the path is absolute or relative. If absolute, start from root.
		if (directory[0] == '/') {
			prev = 1;
		}

		while ((pos = directory.find_first_of('/', prev)) != std::string::npos) {
			dir = directory.substr(0, pos++);
			prev = pos;
			if (dir.size() == 0) continue; // if leading /, first dir is empty string
			mdret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (mdret == -1) {
				if (errno != EEXIST) {
					throw std::runtime_error("Could not create directory");
					//return false; // if the error is not because the directory exists, return false
				}
			}
		}

		// Try to create the last segment of the path.
		mdret = mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (mdret == -1) {
			if (errno != EEXIST) {
				throw std::runtime_error("Could not create directory");
				//return false;
			}
		}
	}
	

	std::vector<std::string> getFilenameInDirectory(const char* dirname, int flag) {

		std::vector<std::string> filenames;
		DIR* dir = opendir(dirname);
		if (dir != nullptr) {
			struct dirent* entry;
			while ((entry = readdir(dir)) != nullptr) {
				// Skip the special entries "." and ".."
				if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
					if (flag == 1) { // keep directory
						if (entry->d_type == DT_DIR)
							filenames.push_back(entry->d_name);
					}
					else if (flag == 2) { // keep file
						if (entry->d_type == DT_REG)
							filenames.push_back(entry->d_name);				
					}
					else {
						filenames.push_back(entry->d_name);
					}
				}
			}
			closedir(dir);
		}
		std::sort(filenames.begin(),filenames.end());
		return filenames;
	}

	
#endif




}