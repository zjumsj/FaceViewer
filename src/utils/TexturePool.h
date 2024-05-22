#pragma once

#include "NImage.h"
#include <map>
#include <string>

// #ifdef _WIN32
// #define OBJMESH_ENABLE_GL
// #endif

class TexturePool
{
public:
	typedef RgbaImage Image;
	struct Texture
	{
		std::string filename;
		Image image;
//#ifdef OBJMESH_ENABLE_GL
		unsigned int texture_id;
		Texture() :texture_id(0) {}
//#endif
	};
public:
	static Texture queryTexture(const std::string& filename);

	// in case that the image file is changed in the disk, we may need to manually update it
	static Texture updateTexture(const std::string& filename);

	static bool containsTexture(const std::string& filename);

	static void resetPool();

	static void removeTexture(const std::string& filename);

	static bool saveTexture(const std::string& filename, const Texture& texture);
private:
	static std::map<std::string, Texture> s_imagePool;
};

