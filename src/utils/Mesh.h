#pragma once
#include <Eigen/Eigen>
#include <vector>

#include "TexturePool.h"

class Mesh {

public:
	const static int MAX_VERT_COUNT = 4;
	const static int MATERIAL_NAME_SIZE = 255;
	const static int OBJ_LINE_SIZE = 500;
	const static int OBJ_FILENAME_LENGTH = 500;
	struct obj_face
	{
		int vertex_index[MAX_VERT_COUNT];
		int normal_index[MAX_VERT_COUNT];
		int texture_index[MAX_VERT_COUNT];
		int vertex_count;
		int material_index;
	};
	struct obj_material
	{
		char name[MATERIAL_NAME_SIZE];
		//TexturePool::Texture texture;
		Eigen::Vector3f amb;
		Eigen::Vector3f diff;
		Eigen::Vector3f spec;
		float  reflect;
		float  refract;
		float  trans;
		float  shiny;
		float  glossy;
		float  refract_index;
		obj_material()
		{
			amb[0] = 0.0f;
			amb[1] = 0.0f;
			amb[2] = 0.0f;
			diff[0] = 0.8f;
			diff[1] = 0.8f;
			diff[2] = 0.8f;
			spec[0] = 0.0f;
			spec[1] = 0.0f;
			spec[2] = 0.0f;
			reflect = 0.0f;
			refract = 0.0f;
			trans = 1;
			glossy = 98;
			shiny = 0;
			refract_index = 1;
		}
		//void drawMat(int isTextureEnabled)const;
		//void generateTextures();
	};

public:
	std::vector<Eigen::Vector3f> vertex_list; //vertex list
	std::vector<Eigen::Vector3f> vertex_normal_list;
	std::vector<Eigen::Vector3f> vertex_color_list;
	std::vector<Eigen::Vector3f> face_normal_list;
	std::vector<Eigen::Vector2f> vertex_texture_list;
	std::vector<obj_face> face_list; // face info
	std::vector<std::vector<int>> vertex_2_face;//
	//std::vector<int> landMark3DIds; //landMark3DIds
	//std::vector<int> subMesh3DIds;  //subMesh3DIds
	//std::vector<int> mesh_2_submesh;
	//
	//std::vector<int> vertex_is_landmark;
	//std::vector<int> vertex_is_submesh;//

	std::vector<Eigen::Vector4f> vertex_extra_list;

public:
	Mesh();
	~Mesh();
	int loadObj(const char * filename, bool isNormalGen, bool isNormalize);
	void scale(float mult);
	void offset(float * v3);
	void updateVertex2Face();
	void updateNormals();
	//bool loadLandMarks(const char * filename);
	//bool loadSubMesh(const char * filename);
	//void updateMesh2Submesh();
	//void buildIndex();

	// The normal process
	// loadObj + scale
	// loadLandMarks + loadSubMesh
	// updateVertex2Face + updateMesh2Submesh
	// buildIndex

	////////////////////////
	void clear();
	void Render(int mesh_param); 
	void RenderShader(int normal = -1, int color = -1, int texture = -1, int extra = -1);
	bool saveObj(const char * filename) const;
	void flatten(bool smooth = true) const;// need manual flatten in this tool :)

	Eigen::Vector3f getOffset(unsigned char do_modify = 0x00);
	void getBoundingBox(Eigen::Vector3f & min_bound, Eigen::Vector3f & max_bound);
	
	// transform obj content
	static bool transObj(const char * loadfile, const char * savefile, float * trans16, float * norm9 = nullptr);

	bool saveObj_debug1(const char* filename) const;
	bool saveObj_debug2(const char* filename) const; // split quads to triangles
	void flipFace();

protected:
	int obj_parse_vertex_index(int * vertex_index, int * texture_index, int * normal_index) const;

	mutable float * _fast_vertex;
	mutable float * _fast_normal;
	mutable float * _fast_color;
	mutable float * _fast_texture;
	mutable int _fast_n_triangle;

	mutable float* _fast_extra;

public:
	char scene_filename[OBJ_FILENAME_LENGTH];

protected:
	std::string _name;

};


void GenerateCube(Mesh & mesh, float scale = 1.f, bool generate_tex_coord = true);
void GenerateCylinder(Mesh & mesh, int N, float scaleR = 1.f, float scaleH = 1.f, bool generate_tex_coord = true);

///////////////////////////////////


class SMesh {
	//RENDER OPTIONS
public:
	const static int SW_V = 0x00000001;
	const static int SW_E = 0x00000002;
	const static int SW_F = 0x00000004;
	const static int SW_N = 0x00000008;
	const static int SW_FLAT = 0x00000010;
	const static int SW_SMOOTH = 0x00000020;
	const static int SW_TEXTURE = 0x00000040;
	const static int SW_LIGHTING = 0x00000080;

public:
	const static int MAX_VERT_COUNT = 4;
	const static int MATERIAL_NAME_SIZE = 255;
	const static int OBJ_FILENAME_LENGTH = 500;
	const static int OBJ_LINE_SIZE = 500;

	struct obj_face
	{
		int vertex_index[MAX_VERT_COUNT];
		int normal_index[MAX_VERT_COUNT];
		int texture_index[MAX_VERT_COUNT];
		int vertex_count;
		int material_index;
	};

	struct obj_material
	{
		char name[MATERIAL_NAME_SIZE];
		TexturePool::Texture texture;
		Eigen::Vector3f amb;//ambient color
		Eigen::Vector3f diff;//diffuse color
		Eigen::Vector3f spec;//specular color
		int illum; //0-10  -1:undetermined
		float  reflect;
		float  refract;
		float  trans;//transparent
		float  shiny;//cos(theta)^shiny
		float  glossy;//sharpness ?
		float  refract_index;// Ni optical density air=1.0

		obj_material();
		void drawMat(int isTextureEnabled)const;
		void generateTextures();
	};

	// Two usage
	// I: group_begin group_end
	// II: smooth_group begin_index
	struct obj_pair {
		int first;
		int second;
		obj_pair();
		obj_pair(int _first, int _second);
		// group info
		int Begin() const;
		int End() const;
		int Offset() const;
		void SetBegin(int a);
		void SetEnd(int a);
		void SetOffset(int a);
		// smooth group info
		int SmoothGroup() const;
		int Index() const;
		void SetSmoothGroup(int a);
		void SetIndex(int a);
	};

	struct obj_group {
		obj_pair group_info; //(group_begin_index,group_end_index) index of faces
		std::vector<obj_pair> smooth_info; //(smooth_group,begin_index) index of faces 
		void clear();
	};

	struct obj_render {
		std::vector<std::vector<Eigen::Vector3f>> _fast_view_verts;
		std::vector<std::vector<Eigen::Vector3f>> _fast_view_vertex_normals;
		std::vector<std::vector<Eigen::Vector3f>> _fast_view_face_normals;
		std::vector<std::vector<Eigen::Vector2f>> _fast_view_texcoords;
		bool _valid_fast_view;
		obj_render();
		void clear();
	};

	struct obj_boundingbox {
		Eigen::Vector3f minmax[2];
		obj_boundingbox();
		Eigen::Vector3f lowerBound() const;
		Eigen::Vector3f upperBound() const;
	};


public:
	SMesh();
	~SMesh();

	int loadObj(const char * path, bool gennormal = true);
	void saveObj(const char * path, const char * mtlname = NULL);
	void render(int showType, const char * filename = NULL);
	// provide another IO for GLSL shader :TODO test me
	void renderShader(int showType, const char * filename = NULL,
		int sel_material = -1, int normal = -1, int texture = -1
	);
	void clear();

	obj_boundingbox boundingBox(const char * filename = NULL);
	void deBias(const obj_boundingbox * bb = NULL);

	void updateNormals();//updateNormals as smooth group suggests
	void updateNormalsCompulsory();//override all normals

	bool extractObject(SMesh & submesh, const char * filename, bool checkvertex = false, bool checknormal = false, bool checktexture = false);

	void afterPoseChanged(bool update_normal = true); // after we directly modify pos, but topolgy is not modified


	void scale(float v);
	void offset(float * v);

protected:
	int obj_parse_vertex_index(int * vertex_index, int * texture_index, int * normal_index) const;
	int obj_parse_mtl_file(const char * filename);
	void obj_write_mtl_file(const char * filename);
	void updateFaceNormals();
	void renderFaces(const std::vector<int> & group_list_index, int showType, const char * filename = NULL);
	void genFastView(const std::vector<int> & group_list_index, obj_render * render);
	bool needGenNormals(const std::vector<int> & idxsgroup) const;
	bool policyMaking(int m, int N) const;
	void smoothAlgorithmI(const std::vector<int> & idxsgroup, bool forcefill);
	void smoothAlgorithmII(const std::vector<int> & idxsgroup, bool forcefill);
	void roughAlgorithm(const std::vector<int> & idxsgroup, bool forcefill);

public:

	std::string _name;
	char scene_filename[OBJ_FILENAME_LENGTH];
	char material_filename[OBJ_FILENAME_LENGTH];
	static obj_material default_material;
	std::vector< Eigen::Vector3f > vertex_list;
	std::vector<Eigen::Vector3f> vertex_normal_list;
	std::vector<Eigen::Vector2f> vertex_texture_list;
	std::vector<obj_face> face_list;
	std::vector<obj_material> material_list;
	//
	std::vector<Eigen::Vector3f> face_normal_list; bool _valid_face_normal_list;
	//
	std::map<std::string, std::vector<int>> group_map; //string->index
	std::vector<obj_group> group_list;
	//
	obj_render gathering_render;
	std::map<std::string, obj_render> sub_render;

};
