#pragma once


#include <Eigen/Eigen>

#include "Transform3.hpp"

#include <iostream>
#include <fstream>
#include <memory>
#include <float.h>

namespace gaussian_splatting {

	// Define the types and sizes that make up the contents of each Gaussian 
	// in the trained model.
	//typedef sibr::Vector3f Pos;

	typedef Eigen::Vector3f Pos;

	template<int D>
	struct SHs
	{
		float shs[(D + 1) * (D + 1) * 3];
	};
	struct Scale
	{
		float scale[3];
	};
	struct Rot
	{
		float rot[4];
	};

	struct FaceCoord {
		int face_id;
		float barycentric_coord[3];
	};

#pragma pack(push,1)
	struct BasePoint 
	{
		Pos pos;
		float n[3];
		unsigned char color[3];
	};
#pragma pack(pop)

	struct BasePoint2
	{
		Pos pos;
		float color[3];
		float n[3];
	};

	template<int D>
	struct RichPoint
	{
		Pos pos;
		float n[3];
		SHs<D> shs;
		float opacity;
		Scale scale;
		Rot rot;
	};

	float sigmoid(const float m1);
	float inverse_sigmoid(const float m1);

	// TODO, the loader is currently very hacky, A general ply loader is needed ...

	// format in python
	// [x,y,z,nx,ny,nz,red,green,blue], color = uint8
	int loadBasePly(const char* filename,
		std::vector<Pos>& pos,
		std::vector<Eigen::Vector3f>& color,
		std::vector<Eigen::Vector3f>& normal,
		Eigen::Vector3f& minn,
		Eigen::Vector3f& maxx
	);

	// [x,y,z,nx,ny,nz,red,green,blue], color = float32
	int loadBasePly2(const char* filename,
		std::vector<Pos>& pos,
		std::vector<Eigen::Vector3f>& color,
		std::vector<Eigen::Vector3f>& normal,
		Eigen::Vector3f& minn,
		Eigen::Vector3f& maxx
	);


	// format in python
	// [x,y,z,nx,ny,nz,SH3(9*3),opacity,scale(3),rot(4)]
	// f_dc_0
	// f_dc_1
	// f_dc_2
	// f_rest_0
	// f_rest_1
	// f_rest_2
	// ...


	// Load the Gaussians from the given file.
	template<int D>
	int loadPly(const char* filename,
		std::vector<Pos>& pos,
		std::vector<SHs<3>>& shs,
		std::vector<float>& opacities,
		std::vector<Scale>& scales,
		std::vector<Rot>& rot,
		Eigen::Vector3f& minn,
		Eigen::Vector3f& maxx)
	{
		std::ifstream infile(filename, std::ios_base::binary);

		if (!infile.good())
			std::cout << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;
			//SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

		// "Parse" header (it has to be a specific format anyway)
		std::string buff;
		std::getline(infile, buff);
		std::getline(infile, buff);

		std::string dummy;
		std::getline(infile, buff);
		std::stringstream ss(buff);
		int count;
		ss >> dummy >> dummy >> count;

		// Output number of Gaussians contained
		std::cout << "Loading " << count << " Gaussian splats" << std::endl;
		//SIBR_LOG << "Loading " << count << " Gaussian splats" << std::endl;

		while (std::getline(infile, buff))
			if (buff.compare("end_header") == 0)
				break;

		// Read all Gaussians at once (AoS)
		std::vector<RichPoint<D>> points(count);
		infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

		// Resize our SoA data
		pos.resize(count);
		shs.resize(count);
		scales.resize(count);
		rot.resize(count);
		opacities.resize(count);

		// Gaussians are done training, they won't move anymore. Arrange
		// them according to 3D Morton order. This means better cache
		// behavior for reading Gaussians that end up in the same tile 
		// (close in 3D --> close in 2D).
		minn = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
		maxx = -minn;
		for (int i = 0; i < count; i++)
		{
			maxx = maxx.cwiseMax(points[i].pos);
			minn = minn.cwiseMin(points[i].pos);
		}
		std::vector<std::pair<uint64_t, int>> mapp(count);
		for (int i = 0; i < count; i++)
		{
			Eigen::Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
			Eigen::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
			Eigen::Vector3i xyz = scaled.cast<int>();

			uint64_t code = 0;
			// msj: original code , is that correct ? CHECK LATER
			//for (int i = 0; i < 21; i++) {
			//	code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			//	code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			//	code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
			//}
			for (int i = 0; i < 21; i++) {
				code |= ((uint64_t(xyz.x() & (1 << i))) << (3 * i + 0));
				code |= ((uint64_t(xyz.y() & (1 << i))) << (3 * i + 1));
				code |= ((uint64_t(xyz.z() & (1 << i))) << (3 * i + 2));
			}

			mapp[i].first = code;
			mapp[i].second = i;
		}
		auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
			return a.first < b.first;
		};
		std::sort(mapp.begin(), mapp.end(), sorter);

		// Move data from AoS to SoA
		int SH_N = (D + 1) * (D + 1);
		for (int k = 0; k < count; k++)
		{
			int i = mapp[k].second;
			pos[k] = points[i].pos;

			// Normalize quaternion
			float length2 = 0;
			for (int j = 0; j < 4; j++)
				length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
			float length = sqrt(length2);
			for (int j = 0; j < 4; j++)
				rot[k].rot[j] = points[i].rot.rot[j] / length;

			// Exponentiate scale
			for (int j = 0; j < 3; j++)
				scales[k].scale[j] = exp(points[i].scale.scale[j]);

			// Activate alpha
			opacities[k] = sigmoid(points[i].opacity);

			shs[k].shs[0] = points[i].shs.shs[0];
			shs[k].shs[1] = points[i].shs.shs[1];
			shs[k].shs[2] = points[i].shs.shs[2];
			for (int j = 1; j < SH_N; j++)
			{
				shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3]; // 3+(SH_N-1)*i_channel+(j-1)
				shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
				shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
			}
		}
		return count;
	}


	void savePly(const char* filename,
		const std::vector<Pos>& pos,
		const std::vector<SHs<3>>& shs,
		const std::vector<float>& opacities,
		const std::vector<Scale>& scales,
		const std::vector<Rot>& rot,
		const Eigen::Vector3f& minn,
		const Eigen::Vector3f& maxx
	);

	
	///////////////
	// camera 
	

	struct InputCamera {

		typedef std::shared_ptr<InputCamera> Ptr;
		typedef Transform3<float>		Transform3f;

		InputCamera();
		/** Partial constructor
		* \param f focal length in mm
		* \param k1 first distortion parameter
		* \param k2 second distortion parameter
		* \param w  width of input image
		* \param h  height of input image
		* \param id ID of input image
		*/
		InputCamera(float f, float k1, float k2, int w, int h, int id);
		InputCamera(float fy, float fx, float k1, float k2, int w, int h, int id);

		/** \return the camera model matrix (for camera stub rendering for instance). */
		Eigen::Matrix4f			model(void) const { return _transform.matrix(); }

		/** \return the camera view matrix. */
		Eigen::Matrix4f			view(void) const { return _transform.invMatrix(); }

		//Eigen::Matrix4f         proj(void) const {
		//	//if (_isOrtho) {
		//	//	return orthographic(_right, _top, _znear, _zfar);
		//	//}
		//	// else{
		//	return perspective(_fov, _aspect, _znear, _zfar, _p);
		//	//}
		//}

		//Eigen::Matrix4f         viewproj(void) const {
		//	Eigen::Matrix4f matViewProj = proj() * view();
		//	return matViewProj;
		//}

		//bool _isOrtho = false;

		unsigned int _id;  ///< Input camera id

		float _focal; ///< focal length
		float _focalx; ///< focal length x, if there is one (colmap typically; -1 by default use with caution)
		float _k1; ///< K1 bundler distorsion parameter
		float _k2; ///< K2 bundler dist parameter
		unsigned int _w; ///< Image width
		unsigned int _h; ///< Image height
		std::string _name; ///< Input image name
		bool _active; ///< is the camera currently in use.


		Transform3f		_transform; ///< The camera pose.
		float			_fov; ///< The vertical field of view (radians)
		float			_aspect; ///< Aspect ratio.
		float			_znear; ///< Near plane.
		float			_zfar; ///< Far plane.

	//private:
	//	mutable bool    _dirtyViewProj;
	};

	std::vector<InputCamera::Ptr> loadJSON(const std::string& jsonPath, const float zNear, const float zFar);


}