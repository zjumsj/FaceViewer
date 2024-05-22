
#include "io.h"
#include "picojson.hpp"

namespace gaussian_splatting {
	
	float sigmoid(const float m1)
	{
		return 1.0f / (1.0f + exp(-m1));
	}

	float inverse_sigmoid(const float m1)
	{
		return log(m1 / (1.0f - m1));
	}

	int loadBasePly(const char* filename,
		std::vector<Pos>& pos,
		std::vector<Eigen::Vector3f>& color,
		std::vector<Eigen::Vector3f>& normal,
		Eigen::Vector3f& minn,
		Eigen::Vector3f& maxx
	) {
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
		std::cout << "Loading " << count << " Points" << std::endl;
		
		while (std::getline(infile, buff))
			if (buff.compare("end_header") == 0)
				break;

		// Read all Gaussians at once (AoS)
		std::vector<BasePoint> points(count);
		infile.read((char*)points.data(), count * sizeof(BasePoint));

		// Resize our SoA data
		pos.resize(count);
		color.resize(count);
		normal.resize(count);
		//printf("pos0,0 %f\n", points[0].pos[0]);
		//printf("pos1,0 %f\n", points[1].pos[0]);
		//printf("pos2,0 %f\n", points[2].pos[0]);

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
		
		for (int k = 0; k < count; k++) {
			pos[k] = points[k].pos;
			for (int j = 0; j < 3; j++) {
				color[k][j] = float(points[k].color[j])/255.f;
				normal[k][j] = points[k].n[j];
			}		
		}	
		return count;
	}

	int loadBasePly2(const char* filename,
		std::vector<Pos>& pos,
		std::vector<Eigen::Vector3f>& color,
		std::vector<Eigen::Vector3f>& normal,
		Eigen::Vector3f& minn,
		Eigen::Vector3f& maxx
	) {
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
		std::cout << "Loading " << count << " Points" << std::endl;

		while (std::getline(infile, buff))
			if (buff.compare("end_header") == 0)
				break;

		// Read all Gaussians at once (AoS)
		std::vector<BasePoint2> points(count);
		infile.read((char*)points.data(), count * sizeof(BasePoint2));

		// Resize our SoA data
		pos.resize(count);
		color.resize(count);
		normal.resize(count);

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

		for (int k = 0; k < count; k++) {
			pos[k] = points[k].pos;
			for (int j = 0; j < 3; j++) {
				color[k][j] = points[k].color[j];
				normal[k][j] = points[k].n[j];
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
	){
		// Read all Gaussians at once (AoS)
		int count = 0;
		for (int i = 0; i < pos.size(); i++)
		{
			if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
				pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
				continue;
			count++;
		}
		std::vector<RichPoint<3>> points(count);

		// Output number of Gaussians contained
		//SIBR_LOG << "Saving " << count << " Gaussian splats" << std::endl;
		std::cout << "Saving " << count << " Gaussian splats" << std::endl;

		std::ofstream outfile(filename, std::ios_base::binary);

		outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

		std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2" };
		std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

		for (auto s : props1)
			outfile << "property float " << s << std::endl;
		for (int i = 0; i < 45; i++)
			outfile << "property float f_rest_" << i << std::endl;
		for (auto s : props2)
			outfile << "property float " << s << std::endl;
		outfile << "end_header" << std::endl;

		count = 0;
		for (int i = 0; i < pos.size(); i++)
		{
			if (pos[i].x() < minn.x() || pos[i].y() < minn.y() || pos[i].z() < minn.z() ||
				pos[i].x() > maxx.x() || pos[i].y() > maxx.y() || pos[i].z() > maxx.z())
				continue;
			points[count].pos = pos[i];
			points[count].rot = rot[i];
			// Exponentiate scale
			for (int j = 0; j < 3; j++)
				points[count].scale.scale[j] = log(scales[i].scale[j]);
			// Activate alpha
			points[count].opacity = inverse_sigmoid(opacities[i]);
			points[count].shs.shs[0] = shs[i].shs[0];
			points[count].shs.shs[1] = shs[i].shs[1];
			points[count].shs.shs[2] = shs[i].shs[2];
			for (int j = 1; j < 16; j++)
			{
				points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
				points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
				points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
			}
			count++;
		}
		outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
	}

	//////////////////////////////

#define FOCAL_X_UNDEFINED -1

	InputCamera::InputCamera()
		: _focal(0.f), _k1(0.f), _k2(0.f), _w(0), _h(0), _id(0), _active(true)
	{}

	InputCamera::InputCamera(float f, float k1, float k2, int w, int h, int id) :
		_focal(f), _k1(k1), _k2(k2), _w(w), _h(h), _id(id), _active(true), _name(""), _focalx(FOCAL_X_UNDEFINED)
	{
		// Update fov and aspect ratio.
		float fov = 2.0f * atan(0.5f * h / f);
		float aspect = float(w) / float(h);

		//Camera::aspect(aspect);
		//Camera::fovy(fov);
		_aspect = aspect;
		_fov = fov;

		_id = id;
	}

	InputCamera::InputCamera(float fy, float fx, float k1, float k2, int w, int h, int id) :
		_focal(fy), _k1(k1), _k2(k2), _w(w), _h(h), _id(id), _active(true), _name(""), _focalx(fx)
	{
		// Update fov and aspect ratio.
		float fovY = 2.0f * atan(0.5f * h / fy);
		float fovX = 2.0f * atan(0.5f * w / fx);

		//Camera::aspect(tan(fovX / 2) / tan(fovY / 2));
		//Camera::fovy(fovY);
		_aspect = tan(fovX / 2) / tan(fovY / 2);
		_fov = fovY;		

		_id = id;
	}
	
	
	std::vector<InputCamera::Ptr> loadJSON(const std::string& jsonPath, const float zNear, const float zFar)
	{
		std::ifstream json_file(jsonPath, std::ios::in);

		if (!json_file)
		{
			std::cerr << "file loading failed: " << jsonPath << std::endl;
			return std::vector<InputCamera::Ptr>();
		}

		std::vector<InputCamera::Ptr> cameras;

		picojson::value v;
		picojson::set_last_error(std::string());
		std::string err = picojson::parse(v, json_file);
		if (!err.empty()) {
			picojson::set_last_error(err);
			json_file.setstate(std::ios::failbit);
		}

		picojson::array& frames = v.get<picojson::array>();

		for (size_t i = 0; i < frames.size(); ++i)
		{
			int id = frames[i].get("id").get<double>();
			std::string imgname = frames[i].get("img_name").get<std::string>();
			int width = frames[i].get("width").get<double>();
			int height = frames[i].get("height").get<double>();
			float fy = frames[i].get("fy").get<double>();
			float fx = frames[i].get("fx").get<double>();

			InputCamera::Ptr camera = std::make_shared<InputCamera>(InputCamera(fy, fx, 0.0f, 0.0f, width, height, id));

			picojson::array& pos = frames[i].get("position").get<picojson::array>();
			Eigen::Vector3f position(pos[0].get<double>(), pos[1].get<double>(), pos[2].get<double>());

			//position.x() = 0;
			//position.y() = 0;
			//position.z() = 1;

			picojson::array& rot = frames[i].get("rotation").get<picojson::array>();
			Eigen::Matrix3f orientation;
			for (int i = 0; i < 3; i++)
			{
				picojson::array& row = rot[i].get<picojson::array>();
				for (int j = 0; j < 3; j++)
				{
					orientation(i, j) = row[j].get<double>();
				}
			}
			orientation.col(1) = -orientation.col(1);
			orientation.col(2) = -orientation.col(2);
			//orientation = sibr::Matrix3f::Identity();

			//camera->name(imgname);
			//camera->position(position);
			//camera->rotation(sibr::Quaternionf(orientation));
			//camera->znear(zNear);
			//camera->zfar(zFar);
			camera->_name = imgname;
			camera->_transform.position(position);
			camera->_transform.rotation(Eigen::Quaternionf(orientation));
			camera->_znear = zNear;
			camera->_zfar = zFar;
			cameras.push_back(camera);
		}
		return cameras;
	}

	
	

}

