
#include "ImageLoader.h"
#include "FreeImage.h"
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <string>


// a very simple implementation, only for 8 bit images
int loadImage(const char * filename, RgbaImage & img) {

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(filename, 0);
	int flag = 0;
	if (fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(filename);

	// check that the plugin has reading capabilities ...
	if ((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))
	{
		// ok, let's load the file
		FIBITMAP *dib = FreeImage_Load(fif, filename, flag);
		if (dib == 0)
		{
			//LVG_LOG(lvg::LVG_LOG_ERROR, "load image failed: " + filename);
			printf("ERR: load image failed: %s\n", filename);
			return 0;
		}
		const int nChannels = FreeImage_GetInfo(dib)->bmiHeader.biBitCount / 8;
		const int width = FreeImage_GetWidth(dib);
		const int height = FreeImage_GetHeight(dib);

		img.create(width, height);
		//texture.image.create(width, height);

		for (int y = 0; y < height; y++)
		{
			const unsigned char* src = FreeImage_GetScanLine(dib, y);
			//unsigned char* dst = texture.image.rowPtr(y);
			unsigned char * dst = img.rowPtr(y);
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
					//*dst++ = 0;
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
	} // end if suc
	else
	{
		printf("ERR: Unsupported image format\n");
		return 0;
	}
	return 1;

};

int saveImage(const char * filename, const RgbaImage & img)
{
	if (img.empty()) {
		printf("WARN: Empty Image: %s\n",filename);
		return 0;
	}

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFIFFromFilename(filename);

	if (fif == FIF_UNKNOWN)
	{
		printf("WARN: Unsupported Format: %s\n", filename);
		return 0;
	}

	const int width = img.width();
	const int height = img.height();
	const int nChannels = img.channels();

	FIBITMAP *dib = FreeImage_Allocate(width, height, nChannels * 8);

	for (int y = 0; y < height; y++)
	{
		//const unsigned char* src = texture.image.rowPtr(y);
		const unsigned char * src = img.rowPtr(y);
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

	int suc = FreeImage_Save(fif, dib, filename);
	FreeImage_Unload(dib);

	if (!suc)
		printf("ERR: write failed: %s\n", filename);

	return int(suc);

}

void gamma_rgba(const unsigned char * src, float * tar, int n_pixels) {
	const float gamma = 2.2f;
	for (int i = 0; i < n_pixels; i++) {
		tar[i * 4 + 0] = powf(float(src[i * 4 + 0]) / 255.f, gamma);
		tar[i * 4 + 1] = powf(float(src[i * 4 + 1]) / 255.f, gamma);
		tar[i * 4 + 2] = powf(float(src[i * 4 + 2]) / 255.f, gamma);
		tar[i * 4 + 3] = 1.f;
	}
}

void sampleImage(const RgbaImage & img, float * uv, float * rgba, bool is_norm, bool use_gl_coord, int mode)
{
	float fs, ft;
	if (is_norm) {
		fs = uv[0] * (float)img.width();
		ft = uv[1] * (float)img.height();
	}
	else {
		fs = uv[0]; ft = uv[1];
	}
	// FreeImage coord, left bottom as (0,0), the same as openGL, different from openCV
	if (!use_gl_coord) {//flip
		ft = (float)img.height() - ft;
	}
	int iu, iuNext;
	int iv, ivNext;
	float ds, dt;
	float x1[4];
	float x2[4];
	float x3[4];
	float x4[4];
	const unsigned char * src = nullptr;

	if (mode == 0) { // nearest
		iu = (int)floor(fs);
		if (iu < 0) iu = 0;
		if (iu >= img.width()) iu = img.width() - 1;
		iv = (int)floor(ft);
		if (iv < 0) iv = 0;
		if (iv >= img.height()) iv = img.height() - 1;
		src = img.rowPtr(iv) + iu * 4;
		for (int i = 0; i < 4; i++) {
			rgba[i] = float(src[i]);
		}
	}
	else if (mode == 1) { // bilinear
		iu = (int)floor(fs - 0.5f);
		iuNext = iu + 1;
		ds = fs - iu - 0.5f;
		if (iu < 0)  iu = 0;
		if (iu >= img.width()) iu = img.width() - 1;
		if (iuNext < 0) iuNext = 0;
		if (iuNext >= img.width()) iuNext = img.width() - 1;

		iv = (int)floor(ft - 0.5f);
		ivNext = iv + 1;
		dt = ft - iv - 0.5f;
		if (iv < 0) iv = 0;
		if (iv >= img.height()) iv = img.height() - 1;
		if (ivNext < 0) ivNext = 0;
		if (ivNext >= img.height()) ivNext = img.height() - 1;

		src = img.rowPtr(iv) + iu * 4;//(0,0)
		for (int i = 0; i < 4; i++) {
			x1[i] = float(src[i]);
		}

		src = img.rowPtr(iv) + iuNext * 4;//(1,0)
		for (int i = 0; i < 4; i++) {
			x2[i] = float(src[i]);
		}

		for (int i = 0; i < 4; i++) {
			x3[i] = x1[i] * (1.f - ds) + x2[i] * ds;
		}
		//
		src = img.rowPtr(ivNext) + iu * 4;//(0,1)
		for (int i = 0; i < 4; i++) {
			x1[i] = float(src[i]);
		}

		src = img.rowPtr(ivNext) + iuNext * 4;//(1,1)
		for (int i = 0; i < 4; i++) {
			x2[i] = float(src[i]);
		}

		for (int i = 0; i < 4; i++) {
			x4[i] = x1[i] * (1.f - ds) + x2[i] * ds;
		}
		//
		for (int i = 0; i < 4; i++) {
			rgba[i] = x3[i] * (1.f - dt) + x4[i] * dt;
		}
	}
}


static void skipSpacesAndComments(std::fstream& file)
{
	while (true)
	{
		if (isspace(file.peek())) {
			file.ignore();
		}
		else  if (file.peek() == '#') {
			std::string line; std::getline(file, line);
		}
		else break;
	}
}

int loadPFM(const char * filename, RgbaFloatImage & img)
{
	std::fstream file;
	file.exceptions(std::fstream::failbit | std::fstream::badbit);
	file.open(filename, std::fstream::in | std::fstream::binary);

	/* read file type */
	char cty[2]; file.read(cty, 2);
	skipSpacesAndComments(file);
	std::string type(cty, 2);

	/* read width, height, and maximum color value */
	int width; file >> width;
	skipSpacesAndComments(file);
	int height; file >> height;
	skipSpacesAndComments(file);
	float maxColor; file >> maxColor;
	if (maxColor > 0) {
		//THROW_RUNTIME_ERROR("Big endian PFM files not supported");
		printf("Big endian PFM files not supported");
		return -1;
	}
	float rcpMaxColor = -1.0f / float(maxColor);
	file.ignore(); // skip space or return

	/* create image and fill with data */
	//Ref<Image> img = new Image4f(width, height, fileName);
	img.create(width, height);

	/* image in binary format 16 bit */
	if (type == "PF")
	{
		float rgb[3] = { 0.f, 0.f, 0.f };
		for (int y = height - 1; y >= 0; y--) {
			
			float * row_ptr = img.rowPtr(y);
			
			for (int x = 0; x<width; x++) {
				file.read((char*)rgb, sizeof(rgb));
				row_ptr[4 * x + 0] = rgb[0] * rcpMaxColor;
				row_ptr[4 * x + 1] = rgb[1] * rcpMaxColor;
				row_ptr[4 * x + 2] = rgb[2] * rcpMaxColor;
				row_ptr[4 * x + 3] = 1.f;
				//img->set(x, y, Color4(rgb[0] * rcpMaxColor, rgb[1] * rcpMaxColor, rgb[2] * rcpMaxColor, 1.0f));
			}
		}
	}

	/* invalid magic value */
	else {
		//THROW_RUNTIME_ERROR("Invalid magic value in PFM file");
		printf("Invalid magic valud in PFM file");
		return -1;
	}
	//return img;
	return 0;
}


int savePFM(const char * filename, const RgbaFloatImage & img) {
	
	/* open file for writing */
	std::fstream file;
	file.exceptions(std::fstream::failbit | std::fstream::badbit);
	file.open(filename, std::fstream::out | std::fstream::binary);

	/* write file header */
	file << "PF" << std::endl;
	//file << img->width << " " << img->height << std::endl;
	file << img.width() << " " << img.height() << std::endl;
	file << -1.0f << std::endl;

	/* write image */
	float c[3];
	for (int y = img.height() - 1; y >= 0; y--) {
		const float * row_ptr = img.rowPtr(y);
		for (int x = 0; x< img.width(); x++) {
			//const Color4 c = img->get(x, y);
			c[0] = row_ptr[4 * x + 0];
			c[1] = row_ptr[4 * x + 1];
			c[2] = row_ptr[4 * x + 2];
			file.write((char*)&c, 3 * sizeof(float));
		}
	}

	return 0;

}