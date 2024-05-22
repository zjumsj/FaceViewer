#include "gs_face.h"

#include "operators.h"

#include <stdexcept>

namespace gaussian_splatting {

	void gaussian_basic::load(const char* filename) {
		pos = cnpy::npy_load(cat_path(filename, "pos.npy").c_str());
		shs = cnpy::npy_load(cat_path(filename, "shs.npy").c_str());
		scale = cnpy::npy_load(cat_path(filename, "scale.npy").c_str());
		opacity = cnpy::npy_load(cat_path(filename, "opacity.npy").c_str());
		rot = cnpy::npy_load(cat_path(filename, "rot.npy").c_str());
	}

	void gaussian_face::load(const char* filename) {
		pos = cnpy::npy_load(cat_path(filename, "pos.npy").c_str());
		shs = cnpy::npy_load(cat_path(filename, "shs.npy").c_str());
		scale = cnpy::npy_load(cat_path(filename, "scale.npy").c_str());
		opacity = cnpy::npy_load(cat_path(filename, "opacity.npy").c_str());
		rot = cnpy::npy_load(cat_path(filename, "rot.npy").c_str());

		xyz_t = cnpy::npy_load(cat_path(filename, "xyz_t.npy").c_str());
		rot_t = cnpy::npy_load(cat_path(filename, "rot_t.npy").c_str());
		scale_t = cnpy::npy_load(cat_path(filename, "scale_t.npy").c_str());
		opacity_t = cnpy::npy_load(cat_path(filename, "opacity_t.npy").c_str());
		shs_t = cnpy::npy_load(cat_path(filename, "shs_t.npy").c_str());

		if (gaussian_splatting::isValidFile(cat_path(filename, "rot_d.npy").c_str()))
		{
			rot_d = cnpy::npy_load(cat_path(filename, "rot_d.npy").c_str());
		}
		pos_t = cnpy::npy_load(cat_path(filename, "pos_t.npy").c_str());
		W = cnpy::npy_load(cat_path(filename, "W.npy").c_str());
		eyelid = cnpy::npy_load(cat_path(filename, "eyelid.npy").c_str());
		//printf("pre features %lld\n", shs.shape[0]);
	}
	void gaussian_face::load_simple(const char* filename) {
		pos = cnpy::npy_load(cat_path(filename, "pos.npy").c_str());
		shs = cnpy::npy_load(cat_path(filename, "shs.npy").c_str());
		scale = cnpy::npy_load(cat_path(filename, "scale.npy").c_str());
		opacity = cnpy::npy_load(cat_path(filename, "opacity.npy").c_str());
		rot = cnpy::npy_load(cat_path(filename, "rot.npy").c_str());

		xyz_t = cnpy::npy_load(cat_path(filename, "xyz_t.npy").c_str());
		rot_t = cnpy::npy_load(cat_path(filename, "rot_t.npy").c_str());
		scale_t = cnpy::npy_load(cat_path(filename, "scale_t.npy").c_str());
		opacity_t = cnpy::npy_load(cat_path(filename, "opacity_t.npy").c_str());
		shs_t = cnpy::npy_load(cat_path(filename, "shs_t.npy").c_str());
	}
	int gaussian_face::getK() const {
		if (shs_t.shape.size() == 0) throw std::runtime_error("Uninitialized!");
		int n_dim = shs_t.shape.size();
		return shs_t.shape[n_dim - 1];
	}

	void transfer_blendshape::load(const char* filename) {
		transfer = cnpy::npy_load(filename);
		assert(transfer.shape.size() == 3);
		assert(transfer.shape[0] == 3);
		assert(transfer.shape[1] == 4);
	}

	int transfer_blendshape::getK() const {
		if (transfer.shape.size() == 0) throw std::runtime_error("Uninitialized!");
		int n_dim = transfer.shape.size();
		return transfer.shape[n_dim - 1];
	}

	void transfer_blendshape::compute_transfer(const float* input, float* out) const{
		int n_dim = getK();
		out[0] = 1.f; out[1] = 0.f; out[2] = 0.f; out[3] = 0.f;
		out[4] = 0.f; out[5] = 1.f; out[6] = 0.f; out[7] = 0.f;
		out[8] = 0.f; out[9] = 0.f; out[10] = 1.f; out[11] = 0.f;
		for (int i = 0; i < 12; i++) {
			for (int k = 0; k < n_dim; k++) {
				out[i] += input[k] * transfer.data<float>()[i * n_dim + k];
			}		
		}	
	}

}