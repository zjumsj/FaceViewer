#pragma once

#include <string.h>

template<typename T, int Channels>
class Image
{
public:

	Image()
		:m_data(nullptr), m_datasize(0), m_width(0), m_height(0), m_stride(0)
	{}

	//Copy From Another Image
	Image(const Image & other) {
		m_data = other.m_data;
		m_datasize = other.m_datasize;
		m_width = other.m_width;
		m_height = other.m_height;
		m_stride = other.m_stride;
		if (other.m_data) {
			m_data = (T *) new char[m_stride * m_height];
			m_datasize = m_stride * m_height;
			memcpy(m_data, other.m_data, m_stride *m_height);
		}
	}

	// Hard Copy
	void operator = (const Image & other) {
		//m_data = other.m_data;//NO MODIFICATION
		//m_datasize = other.m_datasize;//NO MODIFICATION
		m_width = other.m_width;
		m_height = other.m_height;
		m_stride = other.m_stride;
		if (other.m_data) {
			int require_length = m_height * m_stride;
			if (m_datasize < require_length) {//assign new memory
				if (m_data)
					delete[] m_data;
				m_data = (T *)new char[require_length];
				m_datasize = require_length;
			}
			memcpy(m_data, other.m_data, require_length);
		}
	}

	void setConstant(T val) {
		char * ptr = (char *)m_data;
		for (int irow = 0; irow < m_height; irow++) {
			T * ptr_row = (T *)ptr;
			for (int icol = 0; icol < m_width; icol++) {
				for (int ic = 0; ic < Channels; ic++) {
					*ptr_row++ = val;
				}
			}
			ptr += m_stride;
		}
	}

	int width() const { return m_width; }
	int height() const { return m_height; }
	int stride() const { return m_stride; }
	int channels() const { return Channels; }

	//Release all the occupied memory
	void release() {
		if (m_data)
			delete[] m_data;
		m_data = nullptr;
		m_datasize = 0;
		m_width = 0;
		m_height = 0;
		m_stride = 0;
	}

	bool empty() const {
		return (m_width == 0) || (m_height == 0);
	}

	~Image() {
		release();
	}

	void create(int w, int h) {
		m_width = w;
		m_height = h;
		m_stride = w * Channels * sizeof(T);
		m_stride = (m_stride + 3)  / 4 * 4;
		int require_length = m_height * m_stride;
		if (m_datasize < require_length) {//reassign
			if (m_data)
				delete[] m_data;
			m_data = (T *)new char[require_length];
			m_datasize = require_length;
		}
	}

	T* rowPtr(int row) { return (T*)(((char*)m_data) + row*m_stride); }
	const T* rowPtr(int row) const { return (T*)(((char*)m_data) + row*m_stride); }
	T* data() { return m_data; }
	const T* data() const { return m_data; }

private:
	T * m_data;
	int m_datasize;//byte
	int m_width;//pixel
	int m_height;//pixel
	int m_stride;//byte
};

typedef Image<unsigned char, 1> ByteImage;
typedef Image<unsigned char, 3> RgbImage;
typedef Image<unsigned char, 4> RgbaImage;
typedef Image<int, 1> IntImage;
typedef Image<float, 1> FloatImage;
typedef Image<float, 3> RgbFloatImage;
typedef Image<float, 4> RgbaFloatImage;