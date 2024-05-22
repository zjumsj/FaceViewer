#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ static constexpr float kSqrt03_02 = 1.22474487139158894f;
__device__ static constexpr float kSqrt01_03 = 0.57735026918962573f;
__device__ static constexpr float kSqrt02_03 = 0.81649658092772603f;
__device__ static constexpr float kSqrt04_03 = 1.15470053837925146f;
__device__ static constexpr float kSqrt01_04 = 0.50000000000000000f;
__device__ static constexpr float kSqrt03_04 = 0.86602540378443860f;
__device__ static constexpr float kSqrt01_05 = 0.44721359549995793f;
__device__ static constexpr float kSqrt03_05 = 0.77459666924148340f;
__device__ static constexpr float kSqrt06_05 = 1.09544511501033215f;
__device__ static constexpr float kSqrt08_05 = 1.26491106406735176f;
__device__ static constexpr float kSqrt09_05 = 1.34164078649987384f;
__device__ static constexpr float kSqrt05_06 = 0.91287092917527690f;
__device__ static constexpr float kSqrt01_06 = 0.40824829046386302f;
__device__ static constexpr float kSqrt03_08 = 0.61237243569579447f;
__device__ static constexpr float kSqrt05_08 = 0.79056941504209488f;
__device__ static constexpr float kSqrt07_08 = 0.93541434669348533f;
__device__ static constexpr float kSqrt09_08 = 1.06066017177982119f;
__device__ static constexpr float kSqrt05_09 = 0.74535599249992990f;
__device__ static constexpr float kSqrt08_09 = 0.94280904158206336f;

__device__ static constexpr float kSqrt01_10 = 0.31622776601683794f;
__device__ static constexpr float kSqrt03_10 = 0.54772255750516607f;
__device__ static constexpr float kSqrt01_12 = 0.28867513459481287f;
__device__ static constexpr float kSqrt04_15 = 0.51639777949432220f;
__device__ static constexpr float kSqrt01_16 = 0.25000000000000000f;
__device__ static constexpr float kSqrt07_16 = 0.66143782776614768f;
__device__ static constexpr float kSqrt15_16 = 0.96824583655185426f;
__device__ static constexpr float kSqrt01_18 = 0.23570226039551584f;
__device__ static constexpr float kSqrt03_25 = 0.34641016151377546f;
__device__ static constexpr float kSqrt14_25 = 0.74833147735478833f;
__device__ static constexpr float kSqrt15_25 = 0.77459666924148340f;
__device__ static constexpr float kSqrt18_25 = 0.84852813742385702f;
__device__ static constexpr float kSqrt01_32 = 0.17677669529663689f;
__device__ static constexpr float kSqrt03_32 = 0.30618621784789724f;
__device__ static constexpr float kSqrt15_32 = 0.68465319688145765f;
__device__ static constexpr float kSqrt21_32 = 0.81009258730098255f;
__device__ static constexpr float kSqrt01_50 = 0.14142135623730950f;
__device__ static constexpr float kSqrt03_50 = 0.24494897427831780f;
__device__ static constexpr float kSqrt21_50 = 0.64807406984078597f;
__device__ static constexpr float kSqrt1_60 = 0.12909944487358055f;

__device__ void Construct_SH_Rotation_Matrix(const float* mat,
	float sh1[3][3],
	float sh2[5][5],
	float sh3[7][7]
) {
    // level 1
	sh1[0][0] = mat[4]; sh1[0][1] = mat[5]; sh1[0][2] = mat[3];
	sh1[1][0] = mat[7]; sh1[1][1] = mat[8]; sh1[1][2] = mat[6];
	sh1[2][0] = mat[1]; sh1[2][1] = mat[2]; sh1[2][2] = mat[0];

    // level 2
    sh2[0][0] = kSqrt01_04 * ((sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2]) + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2]));
    sh2[0][1] = (sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0]);
    sh2[0][2] = kSqrt03_04 * (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1]);
    sh2[0][3] = (sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2]);
    sh2[0][4] = kSqrt01_04 * ((sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0]) + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]));

    sh2[1][0] = kSqrt01_04 * ((sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2]) + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2]));
    sh2[1][1] = sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0];
    sh2[1][2] = kSqrt03_04 * (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1]);
    sh2[1][3] = sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2];
    sh2[1][4] = kSqrt01_04 * ((sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0]) + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]));

    sh2[2][0] = kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) + -kSqrt01_12 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) + (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
    sh2[2][1] = kSqrt04_03 * sh1[1][1] * sh1[1][0] + -kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0]);
    sh2[2][2] = sh1[1][1] * sh1[1][1] + -kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1]);
    sh2[2][3] = kSqrt04_03 * sh1[1][1] * sh1[1][2] + -kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2]);
    sh2[2][4] = kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) + -kSqrt01_12 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) + (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

    sh2[3][0] = kSqrt01_04 * ((sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2]) + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2]));
    sh2[3][1] = sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0];
    sh2[3][2] = kSqrt03_04 * (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1]);
    sh2[3][3] = sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2];
    sh2[3][4] = kSqrt01_04 * ((sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0]) + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]));

    sh2[4][0] = kSqrt01_04 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
    sh2[4][1] = (sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0]);
    sh2[4][2] = kSqrt03_04 * (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1]);
    sh2[4][3] = (sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2]);
    sh2[4][4] = kSqrt01_04 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

    // level 3
    sh3[0][0] = kSqrt01_04 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
    sh3[0][1] = kSqrt03_02 * (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0]);
    sh3[0][2] = kSqrt15_16 * (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1]);
    sh3[0][3] = kSqrt05_06 * (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2]);
    sh3[0][4] = kSqrt15_16 * (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3]);
    sh3[0][5] = kSqrt03_02 * (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4]);
    sh3[0][6] = kSqrt01_04 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

    sh3[1][0] = kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][0] + sh1[2][0] * sh2[1][4]) + (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4]));
    sh3[1][1] = sh1[1][1] * sh2[0][0] + (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0]);
    sh3[1][2] = kSqrt05_08 * sh1[1][1] * sh2[0][1] + kSqrt05_08 * (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1]);
    sh3[1][3] = kSqrt05_09 * sh1[1][1] * sh2[0][2] + kSqrt05_09 * (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2]);
    sh3[1][4] = kSqrt05_08 * sh1[1][1] * sh2[0][3] + kSqrt05_08 * (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3]);
    sh3[1][5] = sh1[1][1] * sh2[0][4] + (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4]);
    sh3[1][6] = kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][4] - sh1[2][0] * sh2[1][0]) + (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]));

    sh3[2][0] = kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) + kSqrt01_05 * (sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) + -kSqrt1_60 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) - (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
    sh3[2][1] = kSqrt08_05 * sh1[1][1] * sh2[1][0] + kSqrt06_05 * sh1[0][1] * sh2[2][0] + -kSqrt01_10 * (sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0]);
    sh3[2][2] = sh1[1][1] * sh2[1][1] + kSqrt03_04 * sh1[0][1] * sh2[2][1] + -kSqrt01_16 * (sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1]);
    sh3[2][3] = kSqrt08_09 * sh1[1][1] * sh2[1][2] + kSqrt02_03 * sh1[0][1] * sh2[2][2] + -kSqrt01_18 * (sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2]);
    sh3[2][4] = sh1[1][1] * sh2[1][3] + kSqrt03_04 * sh1[0][1] * sh2[2][3] + -kSqrt01_16 * (sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3]);
    sh3[2][5] = kSqrt08_05 * sh1[1][1] * sh2[1][4] + kSqrt06_05 * sh1[0][1] * sh2[2][4] + -kSqrt01_10 * (sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4]);
    sh3[2][6] = kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) + kSqrt01_05 * (sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) + -kSqrt1_60 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) - (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

    sh3[3][0] = kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) + -kSqrt01_10 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) + (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
    sh3[3][1] = kSqrt09_05 * sh1[1][1] * sh2[2][0] + -kSqrt03_05 * (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0]);
    sh3[3][2] = kSqrt09_08 * sh1[1][1] * sh2[2][1] + -kSqrt03_08 * (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1]);
    sh3[3][3] = sh1[1][1] * sh2[2][2] + -kSqrt01_03 * (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2]);
    sh3[3][4] = kSqrt09_08 * sh1[1][1] * sh2[2][3] + -kSqrt03_08 * (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3]);
    sh3[3][5] = kSqrt09_05 * sh1[1][1] * sh2[2][4] + -kSqrt03_05 * (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4]);
    sh3[3][6] = kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) + -kSqrt01_10 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) + (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

    sh3[4][0] = kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) + kSqrt01_05 * (sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) + -kSqrt1_60 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) + (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
    sh3[4][1] = kSqrt08_05 * sh1[1][1] * sh2[3][0] + kSqrt06_05 * sh1[2][1] * sh2[2][0] + -kSqrt01_10 * (sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0]);
    sh3[4][2] = sh1[1][1] * sh2[3][1] + kSqrt03_04 * sh1[2][1] * sh2[2][1] + -kSqrt01_16 * (sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1]);
    sh3[4][3] = kSqrt08_09 * sh1[1][1] * sh2[3][2] + kSqrt02_03 * sh1[2][1] * sh2[2][2] + -kSqrt01_18 * (sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2]);
    sh3[4][4] = sh1[1][1] * sh2[3][3] + kSqrt03_04 * sh1[2][1] * sh2[2][3] + -kSqrt01_16 * (sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3]);
    sh3[4][5] = kSqrt08_05 * sh1[1][1] * sh2[3][4] + kSqrt06_05 * sh1[2][1] * sh2[2][4] + -kSqrt01_10 * (sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4]);
    sh3[4][6] = kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) + kSqrt01_05 * (sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) + -kSqrt1_60 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) + (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));

    sh3[5][0] = kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) - (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
    sh3[5][1] = sh1[1][1] * sh2[4][0] + (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0]);
    sh3[5][2] = kSqrt05_08 * sh1[1][1] * sh2[4][1] + kSqrt05_08 * (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1]);
    sh3[5][3] = kSqrt05_09 * sh1[1][1] * sh2[4][2] + kSqrt05_09 * (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2]);
    sh3[5][4] = kSqrt05_08 * sh1[1][1] * sh2[4][3] + kSqrt05_08 * (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3]);
    sh3[5][5] = sh1[1][1] * sh2[4][4] + (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4]);
    sh3[5][6] = kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) - (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

    sh3[6][0] = kSqrt01_04 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
    sh3[6][1] = kSqrt03_02 * (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0]);
    sh3[6][2] = kSqrt15_16 * (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1]);
    sh3[6][3] = kSqrt05_06 * (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2]);
    sh3[6][4] = kSqrt15_16 * (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3]);
    sh3[6][5] = kSqrt03_02 * (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4]);
    sh3[6][6] = kSqrt01_04 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));
}

