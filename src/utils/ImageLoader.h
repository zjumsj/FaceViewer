#pragma once
#include "NImage.h"

// return 1 success, 0 fail

int loadImage(const char * filename, RgbaImage & img);
int saveImage(const char * filename, const RgbaImage & img);
void gamma_rgba(const unsigned char * src, float * tar, int n_pixels);

// is_norm = true, uv is in [0,1], otherwise [0,w],[0,h]
// mode = 0 nearest
// mode = 1 bilinear
// border mode, clamp to edge
void sampleImage(const RgbaImage & img, float * uv, float * rgba, bool is_norm = true, bool use_gl_coord = true, int mode = 0);

// TODO, not well tested
int loadPFM(const char * filename, RgbaFloatImage & img);
int savePFM(const char * filename, const RgbaFloatImage & img);
