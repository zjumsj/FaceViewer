#include "rot.h"
#include <math.h>

RotTools::RotTools()
{
	reset();
}

RotTools::~RotTools()
{}

void RotTools::reset()
{
	m_view = Eigen::Quaternion<float>(1.f,0.f,0.f,0.f);
	last_pos << 0.f,0.f,1.f;
	now_pos = last_pos;
	// offset
	offset << 0.f,0.f,0.f;
	lastx = 0;
	lasty = 0;
}

void RotTools::leftbutton_down(int w,int h,int x, int y)
{
	float dx,dy,dz;
	dx = 2.f * x / w - 1.f;
	dy = -2.f * y / h + 1.f;
	float temp = dx * dx + dy * dy;
	if (temp < 1.f)
		dz = std::sqrt(1-temp);
	else{
		dz = 0.f;
		dx = dx / std::sqrt(temp);
		dy = dy / std::sqrt(temp);
	}
	last_pos << dx,dy,dz;
	now_pos = last_pos;
}

void RotTools::leftbutton_up(int w,int h,int x,int y)
{
	float dx,dy,dz;
	dx = 2.f * x/w - 1.f;
	dy = -2.f * y/h + 1.f;
	float temp = dx * dx + dy * dy;
	if (temp < 1.f)
		dz = std::sqrt(1-temp);
	else{
		dz = 0.f;
		dx = dx / std::sqrt(temp);
		dy = dy / std::sqrt(temp);
	}
	now_pos << dx,dy,dz;
	m_view = Eigen::Quaternion<float>::FromTwoVectors(last_pos,now_pos)*m_view;
	last_pos = now_pos;
}

void RotTools::leftbutton_trace(int w,int h,int x,int y)
{
	float dx,dy,dz;
	dx = 2.f * x/w - 1.f;
	dy = -2.f * y /h + 1.f;
	float temp = dx * dx + dy * dy;
	if (temp < 1.f)
		dz = std::sqrt(1-temp);
	else{
		dz = 0.f;
		dx = dx/std::sqrt(temp);
		dy = dy/std::sqrt(temp);
	}
	now_pos << dx,dy,dz;
}

void RotTools::getRot(float * c)
{
	Eigen::Quaternion<float> temp = Eigen::Quaternion<float>::FromTwoVectors(last_pos,now_pos) * m_view;
	Eigen::Matrix3f rot = temp.toRotationMatrix();
	c[0]=rot.data()[0];c[1]=rot.data()[1];c[2]=rot.data()[2];c[3]=0.f;
	c[4]=rot.data()[3];c[5]=rot.data()[4];c[6]=rot.data()[5];c[7]=0.f;
	c[8]=rot.data()[6];c[9]=rot.data()[7];c[10]=rot.data()[8];c[11]=0.f;
	c[12]=0.f;c[13]=0.f;c[14]=0.f;c[15]=1.f;
}

void RotTools::rightbutton_down(int x,int y)
{
	lastx = x; lasty = y;
}

void RotTools::rightbutton_trace(float dx,float dy,int x,int y)
{
	offset.x()+=dx*(x-lastx);
	offset.y()-=dy*(y-lasty);//opposite down
	lastx = x; lasty = y;
}

void RotTools::getTrans(float * c){
	c[0] = offset.x();
	c[1] = offset.y();
	c[2] = offset.z();
}

//rot a little
void RotTools::idleRot(float v, int rot_axis){
	Eigen::Vector3f b0,b1;
	if (rot_axis == 2) {
		b0 << 1.f, 0.f, 0.f;
		b1 << cosf(v), sinf(v), 0.f;
	}
	else if (rot_axis == 1) {
		b0 << 0.f, 0.f, 1.f;
		b1 << sinf(v), 0.f, cosf(v);
	}
	else if (rot_axis == 0) {
		b0 << 0.f, 1.f, 0.f;
		b1 << 0.f, cosf(v), sinf(v);
	}
	m_view = m_view * Eigen::Quaternion<float>::FromTwoVectors(b0,b1);
}

///////////////////////////////

void MatrixTools::eye16(float * out16)
{
	for (int i=0;i<16;i++){
		out16[i]=0.f;
		if(i%5==0)
			out16[i]=1.f;
	}
}

void MatrixTools::trans16(float * in3,float * out16)
{
	eye16(out16);
	out16[12]=in3[0];
	out16[13]=in3[1];
	out16[14]=in3[2];
}

void MatrixTools::cp16(float * in16,float * out16)
{
	for(int i=0;i<16;i++)
		out16[i]=in16[i];
}

void MatrixTools::mult16(float * in16_0,float * in16_1,float * out16)
{
	for(int i=0;i<4;i++){
		for (int j=0;j<4;j++){
			float * c = out16+i*4+j;
			*c=0.f;
			for (int k=0;k<4;k++){
				*c = *c + in16_0[j+4*k] * in16_1[k+i*4];
			}
		}
	}
}

void MatrixTools::mult16vec(float * in16, float * vec4, float * out4) {
	for (int i = 0; i < 4; i++) {
		out4[i] = 0;
		for (int j = 0; j < 4; j++) {
			out4[i] += in16[j * 4 + i] * vec4[j];
		}
	}
}

void MatrixTools::transpose16(float * inout16){
	for (int i=0;i<4;i++){
		for (int j=0;j<i;j++){
			float * a = inout16 + 4*i+j;
			float * b = inout16 + 4*j+i;
			float temp=*a;
			*a = *b;*b=temp;
		}
	}
}

void MatrixTools::transpose9(float * inout9) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < i; j++) {
			float * a = inout9 + 3 * i + j;
			float * b = inout9 + 3 * j + i;
			float temp = *a;
			*a = *b; *b = temp;
		}
	}
}

void MatrixTools::apply_trans16(float * in16,float * inout3){
	float temp[3];
	temp[0] = inout3[0];
	temp[1] = inout3[1];
	temp[2] = inout3[2];
	inout3[0] = temp[0]*in16[0] + temp[1]*in16[4] + temp[2]*in16[8] + in16[12];
	inout3[1] = temp[0]*in16[1] + temp[1]*in16[5] + temp[2]*in16[9] + in16[13];
	inout3[2] = temp[0]*in16[2] + temp[1]*in16[6] + temp[2]*in16[10] + in16[14];
}

void MatrixTools::apply_trans9(float * in9, float * inout3) {
	float temp[3];
	temp[0] = inout3[0];
	temp[1] = inout3[1];
	temp[2] = inout3[2];
	inout3[0] = temp[0] * in9[0] + temp[1] * in9[3] + temp[2] * in9[6];
	inout3[1] = temp[0] * in9[1] + temp[1] * in9[4] + temp[2] * in9[7];
	inout3[2] = temp[0] * in9[2] + temp[1] * in9[5] + temp[2] * in9[8];
}

void MatrixTools::inv_of_unity16(float * in16,float * out16){
	// [R',-R't;0,1]
	out16[0] = in16[0]; out16[1] = in16[4]; out16[2] = in16[8]; out16[3] = 0.f;
	out16[4] = in16[1]; out16[5] = in16[5]; out16[6] = in16[9]; out16[7] = 0.f;
	out16[8] = in16[2]; out16[9] = in16[6]; out16[10] = in16[10]; out16[11] = 0.f;
	out16[12] = -(in16[12]* out16[0]+ in16[13]*out16[4]+ in16[14]* out16[8]);
	out16[13] = -(in16[12]* out16[1]+ in16[13]*out16[5]+ in16[14]* out16[9]);
	out16[14] = -(in16[12]* out16[2]+ in16[13]*out16[6]+ in16[14]* out16[10]);
	out16[15] = 1.f;
}

void MatrixTools::get9from16(float * in16,float * out9){
	int j=0;
	for(int i=0;i<3;i++){
		out9[i*3]=in16[j];
		out9[i*3+1]=in16[j+1];
		out9[i*3+2]=in16[j+2];
		j+=4;
	}
}

void MatrixTools::print16(float * in16){
	for (int i=0;i<4;i++){
		for (int j=0;j<4;j++){
			printf("%.3f ",in16[i+4*j]);
		}
		printf("\n");
	}
}

////////////////////////////
// new added 2019/3

void MatrixTools::norm(float * inout3) {
	double length = sqrt(inout3[0] * inout3[0] + inout3[1] * inout3[1] + inout3[2] * inout3[2]);
	if (length > 0.0) {
		inout3[0] /= length;
		inout3[1] /= length;
		inout3[2] /= length;
	}
}

static void lookat_cross(float * in1, float * in2, float * out) {
	out[0] = in1[1] * in2[2] - in1[2] * in2[1];
	out[1] = in1[2] * in2[0] - in1[0] * in2[2];
	out[2] = in1[0] * in2[1] - in1[1] * in2[0];
}

// TODO: may have numerical problem. 
void MatrixTools::lookat(float * view3, float * up3, float * out16) {
	float up[3];
	float right[3];
	lookat_cross(view3, up3, right); 
	norm(right);// R = ( view x up)norm
	lookat_cross(right, view3, up); // up* = right x view
	out16[0] = right[0]; out16[1] = right[1]; out16[2] = right[2]; out16[3] = 0.f;
	out16[4] = up[0]; out16[5] = up[1]; out16[6] = up[2]; out16[7] = 0.f;
	out16[8] = -view3[0]; out16[9] = -view3[1]; out16[10] = -view3[2]; out16[11] = 0.f;
	out16[12] = out16[13] = out16[14] = 0.f; out16[15] = 1.f;
}

//
void MatrixTools::norm_unity16(float * inout16) {
	norm(inout16);
	norm(inout16 + 4);
	float bak[3] = { inout16[8],inout16[9],inout16[10] };
	inout16[8] = inout16[1] * inout16[6] - inout16[2] * inout16[5];
	inout16[9] = inout16[2] * inout16[4] - inout16[0] * inout16[6];
	inout16[10] = inout16[0] * inout16[5] - inout16[1] * inout16[4];
	float dot = bak[0] * inout16[8] + bak[1] * inout16[9] + bak[2] * inout16[10];
	if (dot >= 0.f)
		norm(inout16 + 8);
	else {
		inout16[8] = -inout16[8];
		inout16[9] = -inout16[9];
		inout16[10] = -inout16[10];
		norm(inout16 + 8);
	}
	inout16[15] = 1.f;
}

void MatrixTools::norm_unity9(float * inout9) {
	norm(inout9);
	norm(inout9 + 3);
	float bak[3] = { inout9[6],inout9[7],inout9[8] };
	inout9[6] = inout9[1] * inout9[5] - inout9[2] * inout9[4];
	inout9[7] = inout9[2] * inout9[3] - inout9[0] * inout9[5];
	inout9[8] = inout9[0] * inout9[4] - inout9[1] * inout9[3];
	float dot = bak[0] * inout9[6] + bak[1] * inout9[7] + bak[2] * inout9[8];
	if (dot >= 0.f)
		norm(inout9 + 6);
	else {
		inout9[6] = -inout9[6];
		inout9[7] = -inout9[7];
		inout9[8] = -inout9[8];
		norm(inout9 + 6);
	}
}

// plane4 = [v0,v1,v2,v3]
// v0 * x + v1 * y + v2 * z + v3 = 0
void MatrixTools::planeflip_mat16(float * plane4, float * out16)
{
	float nx = plane4[0]; float ny = plane4[1]; float nz = plane4[2]; float c = plane4[3];
	float nx2 = nx * nx;
	float ny2 = ny * ny;
	float nz2 = nz * nz;
	float den = nx2 + ny2 + nz2;
	out16[0] = (ny2 + nz2 - nx2)/den; out16[4] = (-2 * nx * ny)/den; out16[8] = (-2 * nx * nz)/den; out16[12] = (-2 * nx * c)/den;
	out16[1] =(-2 * ny * nx)/den ; out16[5] = (nx2+nz2-ny2)/den; out16[9] = (-2*ny*nz)/den; out16[13] = (-2 * ny * c)/den;
	out16[2] = (-2 * nz * nx)/den; out16[6] = (-2 * nz * ny)/den; out16[10] = (nx2+ny2-nz2)/den; out16[14] = (-2 * nz * c)/den;
	out16[3] = 0.f; out16[7] = 0.f; out16[11] = 0.f; out16[15] = 1.f;
}

void MatrixTools::rotplane(float * plane4, float * rot16, float * out4) {
	// the first three
	// R * n
	out4[0] = rot16[0] * plane4[0] + rot16[4] * plane4[1] + rot16[8] * plane4[2];
	out4[1] = rot16[1] * plane4[0] + rot16[5] * plane4[1] + rot16[9] * plane4[2];
	out4[2] = rot16[2] * plane4[0] + rot16[6] * plane4[1] + rot16[10] * plane4[2];
	// c'=c-dot(R*n,t)
	out4[3] = plane4[3] - (out4[0] * rot16[12] + out4[1] * rot16[13] + out4[2] * rot16[14]);
}

void MatrixTools::ortho(float * out16, float half_h, float ratio, float znear, float zfar)
{
	float b = 1.f / half_h;
	float a = b / ratio;
	//float c = 1.f / (zfar - znear);
	//float d = -znear * c;
	float c = 2.f / (-zfar + znear);
	float d = (zfar + znear) / (znear - zfar);
	out16[0] = a; out16[1] = 0.f; out16[2] = 0.f; out16[3] = 0.f;
	out16[4] = 0.f; out16[5] = b; out16[6] = 0.f; out16[7] = 0.f;
	out16[8] = 0.f; out16[9] = 0.f; out16[10] = c; out16[11] = 0.f;
	out16[12] = 0.f; out16[13] = 0.f; out16[14] = d; out16[15] = 1.f;
}

void MatrixTools::ortho(float * out16, const float *min3, const float * max3)
{
	float k[3]; 
	float b[3];
	for (int i = 0; i < 3; i++) {
		k[i] = 2.f / (max3[i] - min3[i]);
		b[i] = (-max3[i] - min3[i]) / (max3[i] - min3[i]);
	}
	out16[0] = k[0]; out16[1] = 0.f; out16[2] = 0.f; out16[3] = 0.f;
	out16[4] = 0.f; out16[5] = k[1]; out16[6] = 0.f; out16[7] = 0.f;
	out16[8] = 0.f; out16[9] = 0.f; out16[10] = k[2]; out16[11] = 0.f;
	out16[12] = b[0]; out16[13] = b[1]; out16[14] = b[2]; out16[15] = 1.f;
}

void MatrixTools::perspective(float * out16, float fovy, float ratio, float znear, float zfar)
{
	const float pi = 3.14159265358979323846f;
	float b = 1.f / tan(fovy * pi / 360.f);
	float a = b / ratio;
	//float c = zfar / (znear - zfar);
	float c = (znear + zfar) / (znear - zfar);
	//float d = c * znear;
	float d = 2 * znear * zfar / (znear - zfar);
	out16[0] = a; out16[1] = 0.f; out16[2] = 0.f; out16[3] = 0.f;
	out16[4] = 0.f; out16[5] = b; out16[6] = 0.f; out16[7] = 0.f;
	out16[8] = 0.f; out16[9] = 0.f; out16[10] = c; out16[11] = -1.f;
	out16[12] = 0.f; out16[13] = 0.f; out16[14] = d; out16[15] = 0.f;
}

