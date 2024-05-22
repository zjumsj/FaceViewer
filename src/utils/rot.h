#include <Eigen/Eigen>

class RotTools{
private:
	Eigen::Vector3f last_pos,now_pos;
	Eigen::Quaternion<float> m_view;
	Eigen::Vector3f offset;int lastx,lasty;

public:

	RotTools();
	~RotTools();

	void reset();

	void leftbutton_down(int w,int h,int x, int y);
	void leftbutton_up(int w,int h,int x,int y);
	void leftbutton_trace(int w,int h,int x,int y);
	void getRot(float * c);//c[16]

	void rightbutton_down(int x,int y);
	void rightbutton_trace(float dx,float dy,int x,int y);
	void getTrans(float * c);//c[3]

	void idleRot(float v, int rot_axis=2);
};

class MatrixTools{
public:
	static void eye16(float * out16);
	static void trans16(float * in3,float * out16);
	static void cp16(float * in16,float * out16);
	static void mult16(float * in16_0 ,float * in16_1,float * out16);
	//
	static void mult16vec(float * in16, float * vec4, float * out4);
	// only when R is unity matrix
	static void inv_of_unity16(float * in16,float * out16); 
	static void transpose16(float * inout16);
	static void transpose9(float * inout9);
	static void apply_trans16(float * in16,float * inout3);
	static void apply_trans9(float *in9, float *inout3);
	static void get9from16(float * in16,float * out9);
	// new added 2019/3
	static void norm(float * inout3);
	static void lookat(float * view3, float * up3, float * out16); // normalize view3 and up3 before passing  camera_coord --out16--> world_coord
   // avoid drift 2019/12
	//this code is not tested, so be careful
	static void norm_unity16(float * inout16);
	static void norm_unity9(float * inout9);

	// plane flip
	static void planeflip_mat16(float * plane4, float * out16);
	// suppose rot16 is unity
	static void rotplane(float * plane4, float * rot16, float * out4);

	// new added 2020/10
	static void ortho(float * out16,float half_h, float ratio, float znear, float zfar); // suppose z_far > z_near, towards -z direction
	static void ortho(float * out16, const float * min3, const float * max3);
	static void perspective(float * out16, float fovy, float ratio,float znear, float zfar); // suppose z_far > z_near > 0

	//debug
	static void print16(float * in16);

};