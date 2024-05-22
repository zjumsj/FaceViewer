#include "TexturePool.h"
#ifdef OBJMESH_ENABLE_GL
//#define FREEGLUT_STATIC
#include <GL/glut.h>
#endif
#include "FreeImage.h"
#pragma comment(lib, "FreeImage.lib")

#include <string>

std::map<std::string, TexturePool::Texture> TexturePool::s_imagePool;


void TexturePool::resetPool()
{ 
	s_imagePool.clear(); 
}

void TexturePool::removeTexture(const std::string& filename)
{
	auto iter = s_imagePool.find(filename);
	if (iter != s_imagePool.end())
		s_imagePool.erase(iter);
}

bool TexturePool::containsTexture(const std::string& filename)
{
	auto iter = s_imagePool.find(filename);
	return (iter != s_imagePool.end());
}

TexturePool::Texture TexturePool::queryTexture(const std::string& filename)
{
	Texture texture;
	auto iter = s_imagePool.find(filename);
	if (iter == s_imagePool.end())
		texture = updateTexture(filename);
	else
		texture = iter->second;
	return texture;
}

TexturePool::Texture TexturePool::updateTexture(const std::string& filename)
{
	Texture texture;
	texture.filename = filename;

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(filename.c_str(), 0);
	int flag = 0;
	if (fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(filename.c_str());
	
	// check that the plugin has reading capabilities ...
	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) 
	{
		// ok, let's load the file
		FIBITMAP *dib = FreeImage_Load(fif, filename.c_str(), flag);
		if (dib == 0) 
		{
			//LVG_LOG(lvg::LVG_LOG_ERROR, "load image failed: " + filename);
			printf("ERR: load image failed: %s\n",filename.c_str());
			return texture;
		}
		const int nChannels = FreeImage_GetInfo(dib)->bmiHeader.biBitCount / 8;
		const int width = FreeImage_GetWidth(dib);
		const int height = FreeImage_GetHeight(dib);

		texture.image.create(width, height);

		for (int y = 0; y < height; y++)
		{
			const unsigned char* src = FreeImage_GetScanLine(dib, y);
			unsigned char* dst = texture.image.rowPtr(y);
			switch (nChannels)
			{
			default:
				break;
			case 1:
				for (int x = 0; x < width; x++)
				{
					*dst++ = *src;
					*dst++ = *src;
					*dst++ = *src;
					*dst++ = *src;
					src++;
				}
				break;
			case 3:
				for (int x = 0; x < width; x++)
				{
					*dst++ = src[2];
					*dst++ = src[1];
					*dst++ = src[0];
					*dst++ = 255;
					src += 3;
				}
				break;
			case 4:
				for (int x = 0; x < width; x++)
				{
					*dst++ = src[2];
					*dst++ = src[1];
					*dst++ = src[0];
					*dst++ = src[3];
					src += 4;
				}
				break;
			}
		} // end for y
		FreeImage_Unload(dib);
#ifdef OBJMESH_ENABLE_GL
		glGenTextures(1, &texture.texture_id);

		if (texture.texture_id == 0) {
			printf("ERR: Get OpenGL contex failed!\n");
			//LVG_LOG(lvg::LVG_LOG_ERROR, "Get OpenGL contex failed!\n");
		}
		glBindTexture(GL_TEXTURE_2D, texture.texture_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
#endif
		s_imagePool.insert(std::pair<std::string, Texture>(filename, texture));
	} // end if suc
	else 
	{
		//LVG_LOG(lvg::LVG_LOG_ERROR, "Unsupported image format");
		printf("ERR: Unsupported image format\n");
		return texture;
	}
	
	return texture;
}

bool TexturePool::saveTexture(const std::string& filename, const Texture& texture)
{
	if (texture.image.empty())
	{
		//LVG_LOG(lvg::LVG_LOG_WARN, "Empty Image: " + filename);
		printf("WARN: Empty Image: %s\n",filename.c_str());
		return false;
	}

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFIFFromFilename(filename.c_str());

	if (fif == FIF_UNKNOWN) 
	{
		//LVG_LOG(lvg::LVG_LOG_WARN, "unsupported format: " + filename);
		printf("WARN: Unsupported Format: %s\n",filename.c_str());
		return false;
	}

	const int width = texture.image.width();
	const int height = texture.image.height();
	const int nChannels = texture.image.channels();

	FIBITMAP *dib = FreeImage_Allocate(width, height, nChannels * 8);

	for (int y = 0; y < height; y++)
	{
		const unsigned char* src = texture.image.rowPtr(y);
		unsigned char* dst = FreeImage_GetScanLine(dib, y);
		switch (nChannels)
		{
		default:
			break;
		case 1:
			for (int x = 0; x < width; x++)
				*dst++ = *src++;
			break;
		case 3:
			for (int x = 0; x < width; x++)
			{
				*dst++ = src[2];
				*dst++ = src[1];
				*dst++ = src[0];
				src += 3;
			}
			break;
		case 4:
			for (int x = 0; x < width; x++)
			{
				*dst++ = src[2];
				*dst++ = src[1];
				*dst++ = src[0];
				*dst++ = src[3];
				src += 4;
			}
			break;
		}
	} // end for y

	int suc = FreeImage_Save(fif, dib, filename.c_str());
	FreeImage_Unload(dib);

	if (!suc)
		printf("ERR: write failed: %s\n",filename.c_str());
		//LVG_LOG(lvg::LVG_LOG_ERROR, "write failed: " + filename);

	return suc;
}