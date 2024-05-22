#pragma once

#include "io.h"
//#include "../cnpy/cnpy.h"
#include "cnpy/cnpy.h"

#include <string>
#include <map>

namespace gaussian_splatting {

	// new pipeline

	struct gaussian_basic {

		cnpy::NpyArray pos; // mean face vertex
		cnpy::NpyArray rot;  // mean face rot 
		cnpy::NpyArray scale; // mean face scale
		cnpy::NpyArray opacity; // mean face opacity
		cnpy::NpyArray shs; // mean face sh

		void load(const char* filename);

	};

	struct gaussian_face {

		cnpy::NpyArray pos; // mean face vertex
		cnpy::NpyArray rot;  // mean face rot 
		cnpy::NpyArray scale; // mean face scale
		cnpy::NpyArray opacity; // mean face opacity
		cnpy::NpyArray shs; // mean face sh

		cnpy::NpyArray xyz_t; // basis vertex offset
		cnpy::NpyArray rot_t; // basis rot
		cnpy::NpyArray scale_t; // basis scale
		cnpy::NpyArray opacity_t; // basis opacity
		cnpy::NpyArray shs_t; // basis sh

		cnpy::NpyArray rot_d; // [optional] rot from blendshape
		cnpy::NpyArray pos_t; // offset from pose params
		cnpy::NpyArray W; // skinning weight
		cnpy::NpyArray eyelid;

		void load(const char* filename);
		void load_simple(const char* filename);
		int getK() const; 

	};

	struct transfer_blendshape {

		cnpy::NpyArray transfer; // 3x4xK

		void load(const char* filename);
		int getK() const;
		// row-major format
		void compute_transfer(const float* input, float* out) const;
	};

}