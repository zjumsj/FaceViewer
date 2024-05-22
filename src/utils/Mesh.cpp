#include "GL/glew.h"
#include "Mesh.h"
//#include <lightvg/LightVG.hpp>
#include <fstream>
//#include <GL/GLU.h>
//#include "GL/glut.h"
#include <float.h>
#include <stdexcept>

#define WHITESPACE " \t\n\r"

typedef unsigned char uchar;


bool Mesh::transObj(const char * loadfile, const char  * savefile, float * trans16, float * norm9)
{

	FILE* obj_file_stream;
	FILE * write_stream;
	int current_material = -1;
	char *current_token = NULL;
	char current_line[OBJ_LINE_SIZE];
	char target_line[OBJ_LINE_SIZE];

	int line_number = 0;
	// open scene
	obj_file_stream = fopen(loadfile, "r");
	if (obj_file_stream == 0)
	{
		printf("ERROR: Error reading file: %s\n", loadfile);
		return false;
	}
	write_stream = fopen(savefile, "w");
	if (write_stream == 0) {
		printf("ERROR: Error write file: %s\n", savefile);
		return false;
	}

	Eigen::Map<Eigen::Matrix4f> v_trans(trans16);
	Eigen::Map<Eigen::Matrix3f> * vn_trans = nullptr;
	if (norm9) {
		vn_trans = new Eigen::Map<Eigen::Matrix3f>(norm9);
	}
	
	//parser loop
	while (fgets(current_line, OBJ_LINE_SIZE, obj_file_stream))
	{
		//strcpy_s(target_line, OBJ_LINE_SIZE, current_line);
		strcpy(target_line, current_line);

		current_token = strtok(current_line, " \t\n\r");// 
		line_number++;

		//skip comments
		if (current_token == NULL || current_token[0] == '#')
		{
			// operation here
		}
		//parse objects
		else if (strcmp(current_token, "v") == 0) //process vertex
		{
			Eigen::Vector4f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			v[3] = 1.f;
			v = v_trans * v;
			sprintf(target_line,"v %f %f %f\n",v[0],v[1],v[2]);
		}
		else if (strcmp(current_token, "vn") == 0) //process vertex normal
		{
			Eigen::Vector3f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			v = (*vn_trans) * v;
			//vertex_normal_list.push_back(v);
			sprintf(target_line, "vn %f %f %f\n", v[0], v[1], v[2]);
		}
		else if (strcmp(current_token, "vt") == 0) //process vertex texture
		{
			//Eigen::Vector2f v;
			//v[0] = (float)atof(strtok(NULL, WHITESPACE));
			//v[1] = (float)atof(strtok(NULL, WHITESPACE));
			// operation here
		}
		else if (strcmp(current_token, "f") == 0) //process face 
		{
			// operation here
		}
		/* Reduced */
		else
		{
			//printf("Unknown command '%s' in scene code at line %i: \"%s\".\n",
			//	current_token, line_number, current_line);
		}
		fputs(target_line, write_stream);
	}

	fclose(obj_file_stream);
	fclose(write_stream);
	return true;
}

Mesh::Mesh() {
	_fast_vertex = nullptr;
	_fast_normal = nullptr;
	_fast_color = nullptr;
	_fast_texture = nullptr;
}

Mesh::~Mesh() {
	delete[] _fast_vertex;
	delete[] _fast_normal;
	delete[] _fast_color;
	delete[] _fast_texture;
}


int Mesh::loadObj(const char * filename, bool isNormalGen, bool isNormalize)
{
	// true false
	clear();

	FILE* obj_file_stream;
	int current_material = -1;
	char *current_token = NULL;
	char current_line[OBJ_LINE_SIZE];
	int line_number = 0;
	// open scene
	obj_file_stream = fopen(filename, "r");
	if (obj_file_stream == 0)
	{
		printf("ERROR: Error reading file: %s\n", filename);
		return 0;
	}

	//get name
	strcpy(scene_filename, filename);
	_name = scene_filename;
	int pos1 = (int)_name.find_last_of("\\");
	if (!(pos1 >= 0 && pos1 < (int)_name.size()))
		pos1 = 0;
	int pos2 = (int)_name.find_last_of("/");
	if (!(pos2 >= 0 && pos2 < (int)_name.size()))
		pos2 = 0;
	int pos = std::max(pos1, pos2);
	if (pos) pos++;
	_name = _name.substr(pos, _name.size());

	//parser loop
	while (fgets(current_line, OBJ_LINE_SIZE, obj_file_stream))
	{
		current_token = strtok(current_line, " \t\n\r");// 
		line_number++;

		//skip comments
		if (current_token == NULL || current_token[0] == '#')
			continue;

		//parse objects
		else if (strcmp(current_token, "v") == 0) //process vertex
		{
			Eigen::Vector3f v;			
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_list.push_back(v);
			const char* tmp_ = strtok(NULL, WHITESPACE);
			if (tmp_) { // vertex may have color attached
				Eigen::Vector3f c;
				c[0] = (float)atof(tmp_);
				c[1] = (float)atof(strtok(NULL, WHITESPACE));
				c[2] = (float)atof(strtok(NULL, WHITESPACE));
				vertex_color_list.push_back(c);
			}
		}
		else if (strcmp(current_token, "vn") == 0) //process vertex normal
		{
			Eigen::Vector3f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_normal_list.push_back(v);
		}
		else if (strcmp(current_token, "vt") == 0) //process vertex texture
		{
			Eigen::Vector2f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_texture_list.push_back(v);
		}
		else if (strcmp(current_token, "f") == 0) //process face 
		{
			int vertex_count;
			obj_face face;
			vertex_count = obj_parse_vertex_index(face.vertex_index, face.texture_index, face.normal_index);
			if (vertex_count > 4) printf("too much vertex per face, get %d\n", vertex_count);
			face.vertex_count = vertex_count;//get how much vertex 
			face.material_index = current_material;
			face_list.push_back(face);
		}
		/* Debug */
		else if (strcmp(current_token, "extra") == 0) {
			Eigen::Vector4f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			v[3] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_extra_list.push_back(v);
		}
		/* Reduced */
		else
		{
			printf("Unknown command '%s' in scene code at line %i: \"%s\".\n",
				current_token, line_number, current_line);
		}
	}

	fclose(obj_file_stream);
	printf("INFO: ObjLoaded nv=%zu nf=%zu\n", vertex_list.size(), face_list.size());
	
	if (isNormalGen || vertex_normal_list.size() == 0)
	{
		updateNormals();
	}

	//vertex_is_landmark.resize(vertex_list.size(), 0);
	//vertex_is_submesh.resize(vertex_list.size(), 0);
	vertex_color_list.resize(vertex_list.size(), Eigen::Vector3f::Constant(0.8f));
	
	/*
	if (isNormalize)
	normalizeModel();
	*/
	//updateBoundingBox();
	return 1;


}

void Mesh::scale(float mult) {
	for (int i = 0; i < vertex_list.size(); i++) {
		Eigen::Vector3f & temp = vertex_list[i];
		temp = temp * mult;
	}
}

void Mesh::offset(float * v3) {
	Eigen::Vector3f v(v3[0], v3[1], v3[2]);
	for (int i = 0; i < vertex_list.size(); i++) {
		Eigen::Vector3f & temp = vertex_list[i];
		temp = temp + v;
	}
	//printf("%f %f %f\n", vertex_list[0].x(), vertex_list[0].y(), vertex_list[0].z());
}

int Mesh::obj_parse_vertex_index(int *vertex_index, int *texture_index, int *normal_index)const
{
	char *temp_str;
	char *token;
	int vertex_count = 0;

	while ((token = strtok(NULL, WHITESPACE)) != NULL)
	{
		if (texture_index != NULL)
			texture_index[vertex_count] = -1;
		if (normal_index != NULL)
			normal_index[vertex_count] = -1;

		vertex_index[vertex_count] = atoi(token) - 1;

		if (strstr(token, "//") != 0)  //normal only
		{
			temp_str = strchr(token, '/');
			temp_str++;
			normal_index[vertex_count] = atoi(++temp_str) - 1;
		}
		else if (strstr(token, "/") != 0)
		{
			temp_str = strchr(token, '/');
			texture_index[vertex_count] = atoi(++temp_str) - 1;

			if (strstr(temp_str, "/") != 0)
			{
				temp_str = strchr(temp_str, '/');
				normal_index[vertex_count] = atoi(++temp_str) - 1;
			}
		}
		vertex_count++;
	}
	return vertex_count;
}

void Mesh::updateNormals() {
	face_normal_list.resize(face_list.size());
	vertex_normal_list.resize(vertex_list.size());
	for (int i = 0; i < (int)vertex_normal_list.size(); i++)
		vertex_normal_list[i] = Eigen::Vector3f::Zero();
	for (int i = 0; i < (int)face_list.size(); i++)
	{
		obj_face &f = face_list[i];
		Eigen::Vector3f v = Eigen::Vector3f::Zero();
		for (int j = 0; j <= f.vertex_count - 3; j++) //neighbor
		{
			int j1 = (j + 1) % f.vertex_count;
			int j2 = (j + 2) % f.vertex_count;
			v += Eigen::Vector3f(vertex_list[f.vertex_index[j1]] - vertex_list[f.vertex_index[j]]).cross(
				vertex_list[f.vertex_index[j2]] - vertex_list[f.vertex_index[j]]);
		}
		for (int j = 0; j < f.vertex_count; j++)
		{
			vertex_normal_list[f.vertex_index[j]] += v; //+V
			f.normal_index[j] = f.vertex_index[j]; //
		}
		if (v.norm() != 0)
			face_normal_list[i] = v.normalized();
	}
	for (int i = 0; i < (int)vertex_normal_list.size(); i++)
	{
		if (vertex_normal_list[i].norm() != 0)
			vertex_normal_list[i].normalize(); // 
	}
	//_fast_view_should_update = true;
}

void Mesh::getBoundingBox(Eigen::Vector3f & min_bound, Eigen::Vector3f & max_bound)
{
	size_t n_vertex = vertex_list.size();
	min_bound << FLT_MAX, FLT_MAX, FLT_MAX;
	max_bound << -FLT_MAX, -FLT_MAX, -FLT_MAX;
	for (size_t ivertex = 0; ivertex < n_vertex; ivertex++) {
		for (int j = 0; j < 3; j++) {
			if (vertex_list[ivertex][j] < min_bound[j])
				min_bound[j] = vertex_list[ivertex][j];
			if (vertex_list[ivertex][j] > max_bound[j])
				max_bound[j] = vertex_list[ivertex][j];
		}
	}
}

Eigen::Vector3f Mesh::getOffset(uchar do_modify)
{
	// 0000 0111
	//       xyz bit

	size_t n_vertex = vertex_list.size();
	Eigen::Vector3f ans(0.f,0.f,0.f);
	for (size_t ivertex = 0; ivertex < n_vertex; ivertex++) {
		ans += vertex_list[ivertex];
	}
	ans /= (float)n_vertex;
	if (do_modify & 0x07) {
		for (size_t ivertex = 0; ivertex < n_vertex; ivertex++) {
			if (do_modify & 0x04)
				vertex_list[ivertex][0] -= ans[0];
			if (do_modify & 0x02)
				vertex_list[ivertex][1] -= ans[1];
			if (do_modify & 0x01)
				vertex_list[ivertex][2] -= ans[2];
		}
	}
	return ans;
}

//bool Mesh::loadLandMarks(const char * filename)
//{
//	// 73 74 is not used
//	std::ifstream landmarkFile(filename, std::ifstream::in);
//	if (landmarkFile.fail())
//	{
//		LVG_LOG(lvg::LVG_LOG_ERROR, (std::string("landmark file not found: ") + filename).c_str());
//		return false;
//	}
//	int total;
//	landmarkFile >> total;
//	int temp;
//	landMark3DIds.clear();
//	for (int i = 0; i < total; i++) {
//		landmarkFile >> temp;
//		if (temp >= 0)
//			landMark3DIds.push_back(temp);
//	}
//	landmarkFile.close();
//	std::vector<int> tmp = landMark3DIds;
//	static const int idxMap[] =
//	{
//		14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
//		21, 22, 23, 24, 25, 26,
//		15, 16, 17, 18, 19, 20,
//		31, 32, 33, 34,
//		27, 28, 29, 30,
//		43, 42, 41, 40, 39, 38, 37, 36, 35,
//		45, 44,
//		52, 51, 50, 49, 48, 47, 46, 57, 56, 55, 54, 53,
//		60, 59, 58, 63, 62, 61,
//		64,
//		72, 71, 70, 69,
//		68, 67, 66, 65,
//		74, 73
//	};
//	for (size_t i = 0; i < tmp.size(); i++)
//		landMark3DIds[i] = tmp[idxMap[i]];
//	return true;
//
//}

//bool Mesh::loadSubMesh(const char * filename) {
//	std::ifstream submeshFile(filename, std::ifstream::in);
//	if (submeshFile.fail()) {
//		LVG_LOG(lvg::LVG_LOG_ERROR, (std::string("submesh file not found: ") + filename).c_str());
//		return false;
//	}
//	int total;
//	submeshFile >> total;
//	int temp;
//	subMesh3DIds.clear();
//	for (int i = 0; i < total; i++) {
//		submeshFile >> temp;
//		if (temp >= 0)
//			subMesh3DIds.push_back(temp);
//	}
//	submeshFile.close();
//	return true;
//}

//void Mesh::updateMesh2Submesh() {
//	mesh_2_submesh.clear();
//	mesh_2_submesh.resize(vertex_list.size(), -1);
//	for (int i = 0; i < subMesh3DIds.size(); i++) {
//		mesh_2_submesh[subMesh3DIds[i]] = i;
//	}
//}

//void Mesh::buildIndex() {
//	for (int i = 0; i < landMark3DIds.size(); i++){
//		int ids = landMark3DIds[i];
//		vertex_is_landmark[ids] = 1;
//	}
//	// set submesh
//	for (int i = 0; i < subMesh3DIds.size(); i++) {
//		int ids = subMesh3DIds[i];
//		vertex_is_submesh[ids] = 1;
//	}
//}

void Mesh::updateVertex2Face() {
	vertex_2_face.clear();
	vertex_2_face.resize(vertex_list.size());
	for (int i = 0; i < face_list.size(); i++) {
		obj_face & temp = face_list[i];
		for (int j = 0; j < temp.vertex_count; j++) {
			vertex_2_face[temp.vertex_index[j]].push_back(i);
		}
	}//exists vertex not adjacent to any faces
	/*
	for (int i = 0; i < vertex_2_face.size(); i++) {
		if (vertex_2_face[i].size() == 0) {
			glVertex3fv(vertex_list[i].data());
			printf("%d ", i);
		}
	}
	printf("\n\n\n");*/
}

void Mesh::clear() {
	vertex_list.clear();
	vertex_normal_list.clear();
	vertex_color_list.clear();
	face_normal_list.clear();
	vertex_texture_list.clear();
	face_list.clear();
	vertex_2_face.clear();

	//landMark3DIds.clear();
	//subMesh3DIds.clear();
	//mesh_2_submesh.clear();
	//vertex_is_landmark.clear();
	//vertex_is_submesh.clear();

	scene_filename[0] = 0;
	_name.clear();

	delete[] _fast_vertex; _fast_vertex = nullptr;
	delete[] _fast_normal; _fast_normal = nullptr;
	delete[] _fast_color; _fast_color = nullptr;
	delete[] _fast_texture; _fast_texture = nullptr;
}

//void SampleCode() {
//	int mode = -1;
//	for (size_t i = 0; i < my_obj.face_list.size(); i++) {
//		const Mesh::obj_face & iface = my_obj.face_list[i];
//		if (iface.vertex_count == 3) {
//			if (mode == 4)
//				glEnd();
//			if (mode != 3)
//				glBegin(GL_TRIANGLES);
//			mode = 3;
//			// glVertex3fv goes here
//
//		}
//		else if (iface.vertex_count == 4) {
//			if (mode == 3)
//				glEnd();
//			if (mode != 4)
//				glBegin(GL_QUADS);
//			mode = 4;
//			// glVertex3fv goes here
//		}
//	}
//	if (mode != -1)
//		glEnd();
//}

void Mesh::RenderShader(int normal, int color, int texture, int extra) {
	
	if (face_list.size() == 0) return;
	if (_fast_vertex == nullptr) return;

	std::vector<int> attrib_idx;
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, _fast_vertex);
	if (normal != -1 && _fast_normal) {
		glEnableVertexAttribArray(normal);
		glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 0, _fast_normal);
		attrib_idx.push_back(normal);
	}
	if (color != -1 && _fast_color) {
		glEnableVertexAttribArray(color);
		glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 0, _fast_color);
		attrib_idx.push_back(color);
	}
	if (texture != -1 && _fast_texture) {
		glEnableVertexAttribArray(texture);
		//glVertexAttribPointer(texture, 3, GL_FLOAT, GL_FALSE, 0, _fast_texture);
		glVertexAttribPointer(texture, 2, GL_FLOAT, GL_FALSE, 0, _fast_texture);
		attrib_idx.push_back(texture);
	}
	if (extra != -1 && _fast_extra) {
		glEnableVertexAttribArray(extra);
		glVertexAttribPointer(extra, 4, GL_FLOAT, GL_FALSE, 0, _fast_extra);
		attrib_idx.push_back(extra);
	}
	glDrawArrays(GL_TRIANGLES, 0, /*(int)face_list.size()*/ _fast_n_triangle * 3);

	for (size_t i = 0; i < attrib_idx.size(); i++)
		glDisableVertexAttribArray(attrib_idx[i]);
	glDisableVertexAttribArray(0);	
}

void Mesh::Render(int mesh_param) {
	
	if (face_list.size() == 0) return;
	if (_fast_vertex == nullptr) return;
	
	glEnableClientState(GL_VERTEX_ARRAY);
	if (_fast_normal)
		glEnableClientState(GL_NORMAL_ARRAY);
	if (_fast_texture)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	glVertexPointer(3, GL_FLOAT, 0, _fast_vertex);
	if (_fast_normal)
		glNormalPointer(GL_FLOAT, 0, _fast_normal);
	if (_fast_texture)
		glTexCoordPointer(2, GL_FLOAT, 0, _fast_texture);
	glDrawArrays(GL_TRIANGLES, 0, /*(int)face_list.size()*/ _fast_n_triangle * 3);

	glDisableClientState(GL_VERTEX_ARRAY);
	if (_fast_normal)
		glDisableClientState(GL_NORMAL_ARRAY);
	if (_fast_texture)
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

}


//void Mesh::Render(int mesh_param) {

	//uchar mesh_params[4]; *((int *)mesh_params) = mesh_param;
	//int nfaces = face_list.size();
	//int nverts = vertex_list.size();
	//const obj_face * faces = &(face_list[0]);
	//const Eigen::Vector3f * normals = &vertex_normal_list[0];
	//const Eigen::Vector3f * vertices = &vertex_list[0];

	//glMatrixMode(GL_MODELVIEW);
	//glPushMatrix();

	//if (mesh_params[1]) {
	//	glEnable(GL_POLYGON_OFFSET_FILL);
	//	glPolygonOffset(1., 1.);
	//}
	////Surface
	//if (mesh_params[0] == 1) {// triangle
	//	glEnable(GL_LIGHTING);
	//	glBegin(GL_TRIANGLES);//draw 
	//	for (int i = 0; i < nfaces; i++)
	//	{
	//		const int* vidx = faces[i].vertex_index;//get index ?
	//		float norm[3];
	//		Eigen::Vector3f x1 = vertices[vidx[1]] - vertices[vidx[0]];
	//		Eigen::Vector3f x2 = vertices[vidx[2]] - vertices[vidx[0]];
	//		norm[0] = x1[1] * x2[2] - x1[2] * x2[1];
	//		norm[1] = x1[2] * x2[0] - x1[0] * x2[2];
	//		norm[2] = x1[0] * x2[1] - x1[1] * x2[0];
	//		float L = sqrt(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]);
	//		norm[0] /= L; norm[1] /= L; norm[2] /= L;//��������һ��
	//		glNormal3fv(norm);
	//		glVertex3fv((const float *)&vertices[vidx[0]]);
	//		glVertex3fv((const float *)&vertices[vidx[1]]);
	//		glVertex3fv((const float *)&vertices[vidx[2]]);
	//	}
	//	glEnd();
	//	glDisable(GL_LIGHTING);
	//}
	//else if (mesh_params[0] == 2) {
	//	glEnable(GL_LIGHTING);
	//	glBegin(GL_TRIANGLES);//draw 
	//	for (int i = 0; i < nfaces; i++)
	//	{
	//		const int* vidx = faces[i].vertex_index;//get index ?
	//		glNormal3fv((const float *)&normals[vidx[0]]);
	//		glVertex3fv((const float *)&vertices[vidx[0]]);

	//		glNormal3fv((const float *)&normals[vidx[1]]);
	//		glVertex3fv((const float *)&vertices[vidx[1]]);

	//		glNormal3fv((const float *)&normals[vidx[2]]);
	//		glVertex3fv((const float *)&vertices[vidx[2]]);

	//	}
	//	glEnd();
	//	glDisable(GL_LIGHTING);
	//}
	////Structure
	//if (mesh_params[1] == 1) {

	//	glBegin(GL_POINTS);//draw point
	//	glColor3f(0.6f, 0.5f, 0.4f);
	//	for (int i = 0; i < nverts; i++)
	//	{
	//		if (vertex_is_landmark[i]) {
	//			glColor3f(1, 1, 0);
	//		}
	//		else if (vertex_is_submesh[i]) {
	//			glColor3f(0.7, 0.3, 0.4);
	//		}
	//		else {
	//			glColor3f(0.2, 0.3, 0.6);
	//		}
	//		glVertex3fv((const float *)&vertex_list[i]);
	//	}
	//	glEnd();
	//}
	//else if (mesh_params[1] == 2) { // line mesh
	//	glBegin(GL_LINES);
	//	for (int i = 0; i < nfaces; i++)
	//	{
	//		const int* vidx = faces[i].vertex_index;//get index ?
	//		int nfc = faces[i].vertex_count;// array vidx's num
	//		for (int j = 0; j < nfc; j++)
	//		{
	//			if (vertex_is_landmark[vidx[j]])
	//				glColor3f(1.f, 1.f, 0.f);
	//			else if (vertex_is_submesh[vidx[j]])
	//				glColor3f(0.7f, 0.3f, 0.4f);
	//			else
	//				glColor3f(0.2f, 0.3f, 0.6f);
	//			glVertex3fv((const float*)&vertices[vidx[j]]);
	//			if (vertex_is_landmark[vidx[(j + 1) % nfc]])
	//				glColor3f(1.f, 1.f, 0.f);
	//			else if (vertex_is_submesh[vidx[(j + 1) % nfc]])
	//				glColor3f(0.7f, 0.3f, 0.4f);
	//			else
	//				glColor3f(0.2f, 0.3f, 0.6f);
	//			glVertex3fv((const float *)&vertices[vidx[(j + 1) % nfc]]);
	//		}
	//	}
	//	glEnd();
	//}

	//glPopMatrix();
	//glDisable(GL_POLYGON_OFFSET_FILL);

//}


void Mesh::flipFace(){
	
	int n_face = this->face_list.size();
	for (int i_face = 0; i_face < n_face; i_face++) {
		obj_face face_bak = this->face_list[i_face];
		obj_face & tar_face = this->face_list[i_face];
		if (face_bak.vertex_count == 3) {
			// leave node 0 untouched 
			for (int i = 1; i < face_bak.vertex_count; i++) {
				int i_src = i;
				int i_tar = 3 - i_src;
				tar_face.vertex_index[i_tar] = face_bak.vertex_index[i_src];
				tar_face.normal_index[i_tar] = face_bak.normal_index[i_src];
				tar_face.texture_index[i_tar] = face_bak.texture_index[i_src];
			}
		}
		else if (face_bak.vertex_count == 4) {
			// leave node 0 untouched 
			for (int i = 1; i < face_bak.vertex_count; i++) {
				int i_src = i;
				int i_tar = 4 - i_src;
				tar_face.vertex_index[i_tar] = face_bak.vertex_index[i_src];
				tar_face.normal_index[i_tar] = face_bak.normal_index[i_src];
				tar_face.texture_index[i_tar] = face_bak.texture_index[i_src];
			}
		}	
	}
}

bool Mesh::saveObj_debug2(const char* filename) const {
	FILE* pFile = fopen(filename, "w");
	if (!pFile)
		return false;
	fprintf(pFile, "#number of vertices: %d\n", (int)vertex_list.size());
	for (int i = 0; i < vertex_list.size(); i++)
	{
		Eigen::Vector3f v = vertex_list[i];
		fprintf(pFile, "v %f %f %f\n", v[0], v[1], v[2]);
	}
	fprintf(pFile, "#number of normals: %d\n", (int)vertex_normal_list.size());
	for (int i = 0; i < vertex_normal_list.size(); i++)
	{
		Eigen::Vector3f v = vertex_normal_list[i];
		fprintf(pFile, "vn %f %f %f\n", v[0], v[1], v[2]);
	}
	fprintf(pFile, "#number of texcoords: %d\n", (int)vertex_texture_list.size());
	for (int i = 0; i < vertex_texture_list.size(); i++)
	{
		Eigen::Vector2f v = vertex_texture_list[i];
		fprintf(pFile, "vt %f %f\n", v[0], v[1]);
	}
	int last_mat_id = -1;
	for (int i = 0; i < face_list.size(); i++)
	{
		const obj_face& f = face_list[i];
		
		if (f.vertex_count == 3) {
			fprintf(pFile, "f ");
			for (int k = 0; k < f.vertex_count; k++)
			{
				fprintf(pFile, "%d", f.vertex_index[k] + 1);
				bool has_texture = false;
				if (f.texture_index[k] >= 0 && f.texture_index[k] < vertex_texture_list.size()) {
					fprintf(pFile, "/%d", f.texture_index[k] + 1);
					has_texture = true;
				}
				if (f.normal_index[k] >= 0 && f.normal_index[k] < vertex_normal_list.size()) {
					if (!has_texture) {
						fprintf(pFile, "/");
					}
					fprintf(pFile, "/%d", f.normal_index[k] + 1);
				}
				if (k != f.vertex_count - 1)
					fprintf(pFile, " ");
			}
			fprintf(pFile, "\n");
		}
		else if (f.vertex_count == 4) {
			std::vector<std::vector<int>> aset = { {0,1,2},{0,2,3}};
			for (int i = 0; i < aset.size(); i++) {
				fprintf(pFile, "f ");
				std::vector<int> aset_ = aset[i];
				for (int ik = 0; ik < 3; ik++) {
					int k = aset_[ik];
					fprintf(pFile, "%d", f.vertex_index[k] + 1);
					bool has_texture = false;
					if (f.texture_index[k] >= 0 && f.texture_index[k] < vertex_texture_list.size()) {
						fprintf(pFile, "/%d", f.texture_index[k] + 1);
						has_texture = true;
					}
					if (f.normal_index[k] >= 0 && f.normal_index[k] < vertex_normal_list.size()) {
						if (!has_texture) {
							fprintf(pFile, "/");
						}
						fprintf(pFile, "/%d", f.normal_index[k] + 1);
					}
					if (k != f.vertex_count - 1)
						fprintf(pFile, " ");
				}
				fprintf(pFile, "\n");
			}
		}
		else {
			throw std::runtime_error("not supported!");
		}
		
	}
	fclose(pFile);
	return true;
}


bool Mesh::saveObj_debug1(const char* filename) const {
	FILE* pFile = fopen(filename, "w");
	if (!pFile)
		return false;
	fprintf(pFile, "#number of vertices: %d\n", (int)vertex_list.size());
	for (int i = 0; i < vertex_list.size(); i++)
	{
		Eigen::Vector3f v = vertex_list[i];
		fprintf(pFile, "v %f %f %f\n", v[0], v[1], v[2]);
	}

	fprintf(pFile, "#number of texcoords: %d\n", (int)vertex_texture_list.size());
	for (int i = 0; i < vertex_texture_list.size(); i++)
	{
		Eigen::Vector2f v = vertex_texture_list[i];
		fprintf(pFile, "vt %f %f\n", v[0], v[1]);
	}
	fprintf(pFile, "s 1\n");

	int last_mat_id = -1;
	for (int i = 0; i < face_list.size(); i++)
	{
		const obj_face& f = face_list[i];

		fprintf(pFile, "f ");
		for (int k = 0; k < f.vertex_count; k++)
		{
			
			fprintf(pFile, "%d", f.vertex_index[k] + 1);
			if (f.texture_index[k] >= 0 && f.texture_index[k] < vertex_texture_list.size())
				fprintf(pFile, "/%d", f.texture_index[k] + 1);
			//fprintf(pFile, "/");
			//if (f.normal_index[k] >= 0 && f.normal_index[k] < vertex_normal_list.size())
			//	fprintf(pFile, "%d", f.normal_index[k] + 1);
			if (k != f.vertex_count - 1)
				fprintf(pFile, " ");
		}
		fprintf(pFile, "\n");
	}

	fclose(pFile);
	return true;
}

bool Mesh::saveObj(const char * filename) const {
	FILE* pFile = fopen(filename, "w");
	if (!pFile)
		return false;


	fprintf(pFile, "#number of vertices: %d\n", (int)vertex_list.size());
	for (int i = 0; i < vertex_list.size(); i++)
	{
		Eigen::Vector3f v = vertex_list[i];
		fprintf(pFile, "v %f %f %f\n", v[0], v[1], v[2]);
	}
	fprintf(pFile, "#number of normals: %d\n", (int)vertex_normal_list.size());
	for (int i = 0; i < vertex_normal_list.size(); i++)
	{
		Eigen::Vector3f v = vertex_normal_list[i];
		fprintf(pFile, "vn %f %f %f\n", v[0], v[1], v[2]);
	}
	fprintf(pFile, "#number of texcoords: %d\n", (int)vertex_texture_list.size());
	for (int i = 0; i < vertex_texture_list.size(); i++)
	{
		Eigen::Vector2f v = vertex_texture_list[i];
		fprintf(pFile, "vt %f %f\n", v[0], v[1]);
	}

	int last_mat_id = -1;
	for (int i = 0; i < face_list.size(); i++)
	{
		const obj_face & f = face_list[i];

		fprintf(pFile, "f ");
		for (int k = 0; k < f.vertex_count; k++)
		{
			fprintf(pFile, "%d", f.vertex_index[k] + 1);
			bool has_texture = false;
			if (f.texture_index[k] >= 0 && f.texture_index[k] < vertex_texture_list.size()) {
				fprintf(pFile, "/%d", f.texture_index[k] + 1);
				has_texture = true;
			}
			if (f.normal_index[k] >= 0 && f.normal_index[k] < vertex_normal_list.size()) {
				if (!has_texture) {
					fprintf(pFile, "/");
				}
				fprintf(pFile, "/%d", f.normal_index[k] + 1);
			}
			if (k != f.vertex_count - 1)
				fprintf(pFile, " ");
		}
		fprintf(pFile, "\n");
	}

	fclose(pFile);
	return true;

}

void Mesh::flatten(bool smooth) const {

	// clear
	delete[] _fast_vertex; _fast_vertex = nullptr;
	delete[] _fast_normal; _fast_normal = nullptr;
	delete[] _fast_color; _fast_color = nullptr;
	delete[] _fast_texture; _fast_texture = nullptr;
	delete[] _fast_extra; _fast_extra = nullptr;

	int n_triangle = 0;
	int normal_mode = 0;
	if (smooth && vertex_normal_list.size() > 0)
		normal_mode = 2;
	else if (face_normal_list.size() > 0)
		normal_mode = 1;

	for (size_t i_triangle = 0; i_triangle < face_list.size(); i_triangle++) {
		if (face_list[i_triangle].vertex_count == 3) n_triangle++;
		else if (face_list[i_triangle].vertex_count == 4) n_triangle += 2;
	}
	if(vertex_list.size() > 0)
		_fast_vertex = new float[n_triangle * 3*3];
	if(normal_mode > 0)
		_fast_normal = new float[n_triangle * 3*3];
	if(vertex_color_list.size() > 0)
		_fast_color = new float[n_triangle * 3*3];
	if(vertex_texture_list.size() > 0)
		_fast_texture = new float[n_triangle * 3*2];
	if (vertex_extra_list.size() > 0)
		_fast_extra = new float[n_triangle * 3 * 4];
	
	n_triangle = 0;
	for (size_t i_triangle = 0; i_triangle < face_list.size(); i_triangle++) {
		const obj_face & iface = face_list[i_triangle];
		std::vector<int> idx_;
		if (iface.vertex_count == 3) {
			idx_ = { 0,1,2 };
		}
		else if (iface.vertex_count == 4) {
			idx_ = { 0,1,2,0,2,3 };
		}

		for (size_t idx_offset = 0; idx_offset < idx_.size(); idx_offset += 3) {
			// vertex
			for (int i = 0; i < 3; i++) {
				int l_idx = idx_[idx_offset + i];
				const Eigen::Vector3f & v = vertex_list[iface.vertex_index[l_idx]];
				_fast_vertex[n_triangle * 9 + 3 * i + 0] = v[0];
				_fast_vertex[n_triangle * 9 + 3 * i + 1] = v[1];
				_fast_vertex[n_triangle * 9 + 3 * i + 2] = v[2];
			}
			// color
			for (int i = 0; i < 3; i++) {
				int l_idx = idx_[idx_offset + i];
				const Eigen::Vector3f& v = vertex_color_list[iface.vertex_index[l_idx]];
				_fast_color[n_triangle * 9 + 3 * i + 0] = v[0];
				_fast_color[n_triangle * 9 + 3 * i + 1] = v[1];
				_fast_color[n_triangle * 9 + 3 * i + 2] = v[2];
			}
			// extra
			if (_fast_extra) {
				for (int i = 0; i < 3; i++) {
					int l_idx = idx_[idx_offset + i];
					const Eigen::Vector4f & v = vertex_extra_list[iface.vertex_index[l_idx]];
					_fast_extra[n_triangle * 12 + 4 * i + 0] = v[0];
					_fast_extra[n_triangle * 12 + 4 * i + 1] = v[1];
					_fast_extra[n_triangle * 12 + 4 * i + 2] = v[2];
					_fast_extra[n_triangle * 12 + 4 * i + 3] = v[3];
				}
			}
			// normal
			if (normal_mode == 2 && iface.normal_index[0] != -1) {
				for (int i = 0; i < 3; i++) {
					int l_idx = idx_[idx_offset + i];
					const Eigen::Vector3f & v = vertex_normal_list[iface.normal_index[l_idx]];
					_fast_normal[n_triangle * 9 + 3 * i + 0] = v[0];
					_fast_normal[n_triangle * 9 + 3 * i + 1] = v[1];
					_fast_normal[n_triangle * 9 + 3 * i + 2] = v[2];
				}
			}
			else if (normal_mode > 0) {
				for (int i = 0; i < 3; i++) {
					int l_idx = idx_[idx_offset + i];
					const Eigen::Vector3f & v = face_normal_list[i_triangle];
					_fast_normal[n_triangle * 9 + 3 * i + 0] = v[0];
					_fast_normal[n_triangle * 9 + 3 * i + 1] = v[1];
					_fast_normal[n_triangle * 9 + 3 * i + 2] = v[2];
				}
			}

			// texture
			if (iface.texture_index[0] != -1) {
				for (int i = 0; i < 3; i++) {
					int l_idx = idx_[idx_offset + i];
					const Eigen::Vector2f & v = vertex_texture_list[iface.texture_index[l_idx]];
					_fast_texture[n_triangle * 6 + 2 * i + 0] = v[0];
					_fast_texture[n_triangle * 6 + 2 * i + 1] = v[1];
				}
			}
			n_triangle++;
		}
	}
	_fast_n_triangle = n_triangle;
}

////////////////////////////////////////////


SMesh::obj_material SMesh::default_material;

SMesh::SMesh()
{
	clear();
}

SMesh::~SMesh()
{

}

// May be checked to see if new attributions added 
void SMesh::clear()
{
	vertex_list.clear();
	vertex_normal_list.clear();
	vertex_texture_list.clear();
	face_list.clear();
	material_list.clear();

	group_list.clear();
	group_map.clear();

	face_normal_list.clear(); _valid_face_normal_list = false;

	//
	gathering_render.clear();
	sub_render.clear();
}

int SMesh::loadObj(const char * filename, bool gennormal)
{
	clear();

	FILE* obj_file_stream;
	int current_material = -1;
	char *current_token = NULL;
	char current_line[OBJ_LINE_SIZE];
	int line_number = 0;
	int current_smooth_group = 0; // smooth info
	std::vector<std::string> current_group_name; // name stack
	obj_group temp_groupinfo; temp_groupinfo.clear();

	//group_map.insert(std::pair<std::string, std::vector<int>>("default",std::vector<int>()));
	temp_groupinfo.smooth_info.push_back(obj_pair(0, 0));

	// open scene
	obj_file_stream = fopen(filename, "r");
	if (obj_file_stream == 0)
	{
		printf("ERROR: Error reading file: %s\n", filename);
		return 0;
	}

	//get name
	strcpy(scene_filename, filename);
	_name = scene_filename;
	int pos1 = (int)_name.find_last_of("\\");
	if (!(pos1 >= 0 && pos1 < (int)_name.size()))
		pos1 = 0;
	int pos2 = (int)_name.find_last_of("/");
	if (!(pos2 >= 0 && pos2 < (int)_name.size()))
		pos2 = 0;
	int pos = std::max(pos1, pos2);
	if (pos) pos++;
	_name = _name.substr(pos, _name.size());

	//parser loop
	while (fgets(current_line, OBJ_LINE_SIZE, obj_file_stream))
	{
		current_token = strtok(current_line, " \t\n\r");
		line_number++;

		//skip comments
		if (current_token == NULL || current_token[0] == '#')
			continue;

		//parse objects
		else if (strcmp(current_token, "v") == 0) //process vertex
		{
			Eigen::Vector3f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_list.push_back(v);
		}

		else if (strcmp(current_token, "vn") == 0) //process vertex normal
		{
			Eigen::Vector3f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			v[2] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_normal_list.push_back(v);
		}

		else if (strcmp(current_token, "vt") == 0) //process vertex texture
		{
			Eigen::Vector2f v;
			v[0] = (float)atof(strtok(NULL, WHITESPACE));
			v[1] = (float)atof(strtok(NULL, WHITESPACE));
			vertex_texture_list.push_back(v);
		}

		else if (strcmp(current_token, "f") == 0) //process face
		{
			int vertex_count;
			obj_face face;

			vertex_count = obj_parse_vertex_index(face.vertex_index, face.texture_index, face.normal_index);
			face.vertex_count = vertex_count;
			//if (vertex_count > 4) printf("point > 4\n");
			face.material_index = current_material;
			face_list.push_back(face);
		}

		else if (strcmp(current_token, "usemtl") == 0) // usemtl
		{
			char *mtok = strtok(NULL, "\n");
			current_material = -1;
			for (int i = 0; i < (int)material_list.size(); i++)
			{
				int tl = (int)strlen(mtok);
				int tr = (int)strlen(material_list[i].name);
				while (material_list[i].name[tr - 1] == 10) {
					material_list[i].name[tr - 1] = 0;
					tr--;
				}
				if (strncmp(material_list[i].name, mtok, tl) == 0 && tl == tr)
				{
					current_material = i;
				}
			}
		}

		else if (strcmp(current_token, "mtllib") == 0) // mtllib
		{
			strncpy(material_filename, strtok(NULL, WHITESPACE), OBJ_FILENAME_LENGTH);
			std::string fullmat = filename;
			std::string matname = material_filename;

			// relative path
			if (matname.find_first_of(':') == std::string::npos &&
				(matname.find_first_of('/') != 0 || matname.find_first_of('/') == std::string::npos))
			{
				int pos1 = fullmat.find_last_of("\\");
				if (!(pos1 >= 0 && pos1 < (int)fullmat.size()))
					pos1 = 0;
				int pos2 = fullmat.find_last_of("/");
				if (!(pos2 >= 0 && pos2 < (int)fullmat.size()))
					pos2 = 0;
				int pos = std::max(pos1, pos2);
				if (pos) pos++;
				fullmat = fullmat.substr(0, pos);
				fullmat.append(material_filename);
			}
			// absolute path
			else
			{
				fullmat = material_filename;
			}
			//parse mtl file
			obj_parse_mtl_file(fullmat.c_str());
			continue;
		}

		else if (strcmp(current_token, "g") == 0) {//group
			temp_groupinfo.group_info.SetEnd(face_list.size());
			if (temp_groupinfo.group_info.Begin() != temp_groupinfo.group_info.End())
			{	// write info to group
				if (current_group_name.size() == 0)
					current_group_name.push_back(std::string("default"));
				// build index
				for (int iname = 0; iname < current_group_name.size(); iname++) {
					std::map<std::string, std::vector<int>>::iterator it = group_map.find(current_group_name[iname]);
					if (it == group_map.end())
						group_map.insert(std::pair<std::string, std::vector<int>>(current_group_name[iname], std::vector<int>()));
					std::vector<int> & idxes = group_map.find(current_group_name[iname])->second;
					idxes.push_back(group_list.size());
				}
				// build list
				group_list.push_back(temp_groupinfo);
			}
			// set new group
			current_group_name.clear();
			temp_groupinfo.clear();
			temp_groupinfo.group_info.SetBegin(face_list.size()); // current face index
			temp_groupinfo.smooth_info.push_back(obj_pair(current_smooth_group, face_list.size())); // first smooth info
			char * groupname;
			while (groupname = strtok(NULL, WHITESPACE)) {
				current_group_name.push_back(std::string(groupname));
			}
		}

		else if (strcmp(current_token, "s") == 0) //smooth
		{
			char * sindex = strtok(NULL, WHITESPACE);
			int last_smooth_group = current_smooth_group;
			if (strcmp(sindex, "off") == 0)
				current_smooth_group = 0;
			else {
				current_smooth_group = atoi(sindex);
			}
			if (last_smooth_group != current_smooth_group) {// smooth group changed
				const int lastid = temp_groupinfo.smooth_info.size() - 1;
				if (temp_groupinfo.smooth_info[lastid].Index() == face_list.size()) {// change smooth group
					temp_groupinfo.smooth_info[lastid].SetSmoothGroup(current_smooth_group);
				}
				else {// append smooth group
					temp_groupinfo.smooth_info.push_back(obj_pair(current_smooth_group, face_list.size()));
				}
			}
		}

		else
		{
			//printf("Unknown command '%s' in scene code at line %i: \"%s\".\n",
			//	current_token, line_number, current_line);
		}
	}

	fclose(obj_file_stream);

	// write the last group info
	temp_groupinfo.group_info.SetEnd(face_list.size());
	if (temp_groupinfo.group_info.Begin() != temp_groupinfo.group_info.End())
	{	// write info to group
		if (current_group_name.size() == 0)
			current_group_name.push_back(std::string("default"));
		// build index
		for (int iname = 0; iname < current_group_name.size(); iname++) {
			std::map<std::string, std::vector<int>>::iterator it = group_map.find(current_group_name[iname]);
			if (it == group_map.end())
				group_map.insert(std::pair<std::string, std::vector<int>>(current_group_name[iname], std::vector<int>()));
			std::vector<int> & idxes = group_map.find(current_group_name[iname])->second;
			idxes.push_back(group_list.size());
		}
		// build list
		group_list.push_back(temp_groupinfo);
	}

	if (gennormal)
		updateNormals();

	printf("INFO: ObjLoaded nv=%zu nf=%zu\n", vertex_list.size(), face_list.size());

	return 1;

}


// filename give the .obj path and filename
// mtlname give the relative path and filename of the .mtl to obj file or absoluate path and filename
void SMesh::saveObj(const char * filename, const char * mtlname)
{
	FILE* pFile = fopen(filename, "w");
	if (!pFile)
		throw std::runtime_error((std::string("Open file failed: ") + filename).c_str());

	std::string fullname;
	std::string mtlsname;

	if (material_list.size() > 0)
		if (mtlname == NULL)
			fprintf(pFile, "mtllib %s\n", material_filename);
		else { // get full name
			fullname = filename;
			mtlsname = mtlname;
			// relative path
			if (mtlsname.find_first_of(':') == std::string::npos &&
				(mtlsname.find_first_of('/') != 0 || mtlsname.find_first_of('/') == std::string::npos))
			{
				int pos1 = fullname.find_last_of("\\");
				if (!(pos1 >= 0 && pos1 < (int)fullname.size()))
					pos1 = 0;
				int pos2 = fullname.find_last_of("/");
				if (!(pos2 >= 0 && pos2 < (int)fullname.size()))
					pos2 = 0;
				int pos = std::max(pos1, pos2);
				if (pos) pos++;
				fullname = fullname.substr(0, pos);
				fullname.append(mtlsname);
			}
			// absolute path
			else
				fullname = mtlsname;
			fprintf(pFile, "mtllib %s\n", mtlname);
		}

		fprintf(pFile, "#number of vertices: %d\n", (int)vertex_list.size());
		for (int i = 0; i < vertex_list.size(); i++)
		{
			Eigen::Vector3f v = vertex_list[i];
			fprintf(pFile, "v %f %f %f\n", v[0], v[1], v[2]);
		}
		fprintf(pFile, "#number of normals: %d\n", (int)vertex_normal_list.size());
		for (int i = 0; i < vertex_normal_list.size(); i++)
		{
			Eigen::Vector3f v = vertex_normal_list[i];
			fprintf(pFile, "vn %f %f %f\n", v[0], v[1], v[2]);
		}
		fprintf(pFile, "#number of texcoords: %d\n", (int)vertex_texture_list.size());
		for (int i = 0; i < vertex_texture_list.size(); i++)
		{
			Eigen::Vector2f v = vertex_texture_list[i];
			fprintf(pFile, "vt %f %f\n", v[0], v[1]);
		}

		int last_mat_id = -1;
		int curr_group_list_id = 0;
		int curr_smooth_group_list_id = 0;

		std::vector<std::vector<std::string>> name_vector(group_list.size());
		for (std::map<std::string, std::vector<int>>::iterator it = group_map.begin(); it != group_map.end(); it++) {
			std::vector<int> & list_id_vector = it->second;
			for (int i = 0; i < list_id_vector.size(); i++) {
				int list_id = list_id_vector[i];
				name_vector[list_id].push_back(it->first);
			}
		}

		for (int i = 0; i < face_list.size(); i++)
		{
			const SMesh::obj_face& f = face_list[i];

			// group & smooth group
			//if( group_list.size()<=0 ) //Impossible
			if (i >= group_list[curr_group_list_id].group_info.End()) {//go to next
				curr_group_list_id++; curr_smooth_group_list_id = 0;
			}
			if (curr_group_list_id < group_list.size()) {
				obj_group & agroup = group_list[curr_group_list_id];
				obj_pair & smooth_pair = agroup.smooth_info[curr_smooth_group_list_id];
				if (i == smooth_pair.Index()) {//write s info first
					if (smooth_pair.SmoothGroup() == 0)
						fprintf(pFile, "s off\n");
					else
						fprintf(pFile, "s %d\n", smooth_pair.SmoothGroup());
					if (curr_smooth_group_list_id + 1 < agroup.smooth_info.size())//add
						curr_smooth_group_list_id++;
				}
				if (i == agroup.group_info.Begin()) {//begin of a Group
					if (name_vector[curr_group_list_id].size()) {
						fprintf(pFile, "g");
						for (int j = 0; j < name_vector[curr_group_list_id].size(); j++) {
							fprintf(pFile, " %s", name_vector[curr_group_list_id][j].c_str());
						}
						fprintf(pFile, "\n");
					}
				}
			}

			// material 
			if (f.material_index != last_mat_id && f.material_index < material_list.size())
			{
				last_mat_id = f.material_index;
				fprintf(pFile, "usemtl %s\n", material_list[f.material_index].name);
			}

			fprintf(pFile, "f ");
			for (int k = 0; k < f.vertex_count; k++)
			{
				fprintf(pFile, "%d/", f.vertex_index[k] + 1);
				if (f.texture_index[k] >= 0 && f.texture_index[k] < vertex_texture_list.size())
					fprintf(pFile, "%d", f.texture_index[k] + 1);
				fprintf(pFile, "/");
				if (f.normal_index[k] >= 0 && f.normal_index[k] < vertex_normal_list.size())
					fprintf(pFile, "%d", f.normal_index[k] + 1);
				if (k != f.vertex_count - 1)
					fprintf(pFile, " ");
			}
			fprintf(pFile, "\n");
		}

		fclose(pFile);

		if (mtlname &&  material_list.size())
			obj_write_mtl_file(fullname.c_str());
}

// A Naive Implemention
// Better Refer mtl File Format
void SMesh::obj_write_mtl_file(const char * filename)
{
	FILE* pFile = fopen(filename, "w");
	if (!pFile)
		throw std::runtime_error((std::string("Open file failed: ") + filename).c_str());

	for (int im = 0; im < material_list.size(); im++) {
		obj_material & objm = material_list[im];
		fprintf(pFile, "newmtl %s\n", objm.name);
		fprintf(pFile, "Ka %f %f %f\n", objm.amb[0], objm.amb[1], objm.amb[2]);//"%0.3f"
		fprintf(pFile, "Kd %f %f %f\n", objm.diff[0], objm.diff[1], objm.diff[2]);
		fprintf(pFile, "Ks %f %f %f\n", objm.spec[0], objm.spec[1], objm.spec[2]);
		if (objm.illum != -1)
			fprintf(pFile, "illum %d\n", objm.illum);
		//r or ra ?
		fprintf(pFile, "r %f\n", objm.reflect);
		fprintf(pFile, "ra %f\n", objm.refract);
		//
		fprintf(pFile, "d %f\n", 1.f - objm.trans);
		fprintf(pFile, "Ns %f\n", objm.shiny);
		fprintf(pFile, "sharpness %f\n", objm.glossy);
		fprintf(pFile, "Ni %f\n", objm.refract_index);
		if (!objm.texture.filename.empty())
			fprintf(pFile, "map_Kd %s\n", objm.texture.filename.c_str());
	}
	fclose(pFile);
}

int SMesh::obj_parse_vertex_index(int * vertex_index, int * texture_index, int * normal_index) const
{
	char *temp_str;
	char *token;
	int vertex_count = 0;

	while ((token = strtok(NULL, WHITESPACE)) != NULL)
	{
		if (texture_index != NULL)
			texture_index[vertex_count] = -1;
		if (normal_index != NULL)
			normal_index[vertex_count] = -1;

		vertex_index[vertex_count] = atoi(token) - 1;

		if (strstr(token, "//") != 0)  //normal only
		{
			temp_str = strchr(token, '/');
			temp_str++;
			normal_index[vertex_count] = atoi(++temp_str) - 1;
		}
		else if (strstr(token, "/") != 0)
		{
			temp_str = strchr(token, '/');
			texture_index[vertex_count] = atoi(++temp_str) - 1;

			if (strstr(temp_str, "/") != 0)
			{
				temp_str = strchr(temp_str, '/');
				normal_index[vertex_count] = atoi(++temp_str) - 1;
			}
		}

		vertex_count++;
	}

	return vertex_count;
}

int SMesh::obj_parse_mtl_file(const char * filename)
{
	int line_number = 0;
	char *current_token;
	char current_line[OBJ_LINE_SIZE];
	char material_open = 0;
	obj_material *current_mtl = 0;
	FILE *mtl_file_stream;

	// open scene
	mtl_file_stream = fopen(filename, "r");
	if (mtl_file_stream == 0)
	{
		printf("ERROR: Error reading file: %s\n", filename);
		return 0;
	}

	material_list.clear();

	while (fgets(current_line, OBJ_LINE_SIZE, mtl_file_stream))
	{
		current_token = strtok(current_line, " \t\n\r");
		line_number++;

		//skip comments
		if (current_token == NULL || strcmp(current_token, "//") == 0 || strcmp(current_token, "#") == 0)
			continue;


		//start material
		else if (strcmp(current_token, "newmtl") == 0)
		{
			material_open = 1;
			material_list.push_back(obj_material());
			current_mtl = &material_list[material_list.size() - 1];

			// get the name
			strncpy(current_mtl->name, strtok(NULL, "\n"), MATERIAL_NAME_SIZE);
		}

		//ambient
		else if (strcmp(current_token, "Ka") == 0 && material_open)
		{
			current_mtl->amb[0] = (float)atof(strtok(NULL, " \t"));
			current_mtl->amb[1] = (float)atof(strtok(NULL, " \t"));
			current_mtl->amb[2] = (float)atof(strtok(NULL, " \t"));
		}

		//diff
		else if (strcmp(current_token, "Kd") == 0 && material_open)
		{
			current_mtl->diff[0] = (float)atof(strtok(NULL, " \t"));
			current_mtl->diff[1] = (float)atof(strtok(NULL, " \t"));
			current_mtl->diff[2] = (float)atof(strtok(NULL, " \t"));
		}

		//specular
		else if (strcmp(current_token, "Ks") == 0 && material_open)
		{
			current_mtl->spec[0] = (float)atof(strtok(NULL, " \t"));
			current_mtl->spec[1] = (float)atof(strtok(NULL, " \t"));
			current_mtl->spec[2] = (float)atof(strtok(NULL, " \t"));
		}
		//shiny
		else if (strcmp(current_token, "Ns") == 0 && material_open)
		{
			current_mtl->shiny = (float)atof(strtok(NULL, " \t"));
		}
		//transparent
		else if (strcmp(current_token, "Tr") == 0 && material_open)
		{
			current_mtl->trans = (float)atof(strtok(NULL, " \t"));
		}
		else if (strcmp(current_token, "d") == 0 && material_open)
		{
			current_mtl->trans = 1.f - (float)atof(strtok(NULL, " \t"));
		}
		//reflection
		else if (strcmp(current_token, "r") == 0 && material_open)
		{
			current_mtl->reflect = (float)atof(strtok(NULL, " \t"));
		}
		else if (strcmp(current_token, "ra") == 0 && material_open)
		{
			current_mtl->refract = (float)atof(strtok(NULL, " \t"));
		}
		//glossy
		else if (strcmp(current_token, "sharpness") == 0 && material_open)
		{
			current_mtl->glossy = (float)atof(strtok(NULL, " \t"));
		}
		//refract index
		else if (strcmp(current_token, "Ni") == 0 && material_open)
		{
			current_mtl->refract_index = (float)atof(strtok(NULL, " \t"));
		}
		// illumination type
		else if (strcmp(current_token, "illum") == 0 && material_open)
		{
			current_mtl->illum = atoi(strtok(NULL, " \t"));
		}
		// texture map
		else if ((strcmp(current_token, "map_Ka") == 0 ||
			strcmp(current_token, "map_Kd") == 0)
			&& material_open)
		{
			char tmpBuffer[OBJ_FILENAME_LENGTH];
			strncpy(tmpBuffer, strtok(NULL, " \t"), OBJ_FILENAME_LENGTH);
			//remove ' '
			for (int i = (int)strlen(tmpBuffer) - 1; i >= 0; i--)
			{
				if (tmpBuffer[i] != ' ' &&tmpBuffer[i] != '\n')
					break;
				tmpBuffer[i] = 0;
			}

			//to full path
			std::string fullmat = filename;
			std::string texname = tmpBuffer;

			// relative path
			if (texname.find_first_of(':') == std::string::npos &&
				(texname.find_first_of('/') != 0 || texname.find_first_of('/') == std::string::npos))
			{
				int pos1 = fullmat.find_last_of("\\");
				if (!(pos1 >= 0 && pos1 < (int)fullmat.size()))
					pos1 = 0;
				int pos2 = fullmat.find_last_of("/");
				if (!(pos2 >= 0 && pos2 < (int)fullmat.size()))
					pos2 = 0;
				int pos = std::max(pos1, pos2);
				if (pos) pos++;
				fullmat = fullmat.substr(0, pos);
				fullmat.append(tmpBuffer);
			}
			// absolute path
			else
			{
				fullmat = texname;
			}
			current_mtl->texture = TexturePool::queryTexture(fullmat);
		}
		else
		{
			//fprintf(stderr, "Unknown command '%s' in material file %s at line %i:\n\t%s\n",
			//	current_token, filename, line_number, current_line);
			//return 0;
		}
	}
	fclose(mtl_file_stream);

	return 1;

}

void SMesh::updateFaceNormals()
{
	if (_valid_face_normal_list) return;
	const int nfsize = face_list.size() * 2;

	face_normal_list.resize(nfsize);
	for (int i = 0; i < (int)face_list.size(); i++)
	{
		obj_face & f = face_list[i];
		Eigen::Vector3f & v = face_normal_list[2 * i];
		Eigen::Vector3f & v2 = face_normal_list[2 * i + 1];
		if (f.vertex_count == 3) {
			v = (vertex_list[f.vertex_index[1]] - vertex_list[f.vertex_index[0]]).cross(
				vertex_list[f.vertex_index[2]] - vertex_list[f.vertex_index[0]]);
			if (v.norm() != 0)
				v.normalize();
		}
		else if (f.vertex_count == 4) {
			//1 2 3
			//0 1 3
			v = (vertex_list[f.vertex_index[2]] - vertex_list[f.vertex_index[1]]).cross(
				vertex_list[f.vertex_index[3]] - vertex_list[f.vertex_index[1]]);
			if (v.norm() != 0)
				v.normalize();
			//
			v2 = (vertex_list[f.vertex_index[1]] - vertex_list[f.vertex_index[0]]).cross(
				vertex_list[f.vertex_index[3]] - vertex_list[f.vertex_index[0]]);
			if (v2.norm() != 0)
				v2.normalize();
		}
	}

	_valid_face_normal_list = true;
}

bool SMesh::policyMaking(int m, int N) const
{
	// mlogm N
	// 'set' may be a little slower in memory access than 'vector', param should be larger than 1.f
	const float param = 3.f;
	//const float param = 5.f;
	if (m < 1)
		return false;
	return (logf(m)*m*param >(float)N);
}

// have missing normals
bool SMesh::needGenNormals(const std::vector<int> & idxsgroup) const
{
	for (int idx = 0; idx < idxsgroup.size(); idx++) {// O(m)
		const int & iface = idxsgroup[idx];
		const obj_face & f = face_list[iface];
		for (int j = 0; j < f.vertex_count; j++) {
			if (f.normal_index[j] == -1)//the first invalid normal
				return true;
		}
	}
	return false;
}

void SMesh::smoothAlgorithmI(const std::vector<int> & idxsgroup, bool forcefill)
{
	updateFaceNormals();

	const int nV = vertex_list.size();
	std::vector<int> scoreboard(nV);
	std::vector<Eigen::Vector3f> local_vertex_normal_list(nV);
	for (int i = 0; i < nV; i++)// O(N)
		scoreboard[i] = -1;
	for (int idx = 0; idx < idxsgroup.size(); idx++) {//Set Valid O(m)
		const int & iface = idxsgroup[idx];
		obj_face & f = face_list[iface];
		for (int j = 0; j < f.vertex_count; j++) {
			scoreboard[f.vertex_index[j]] = 0;
		}
	}
	for (int i = 0; i < nV; i++) {//Clean Data O(N)
		if (scoreboard[i] == 0)
			local_vertex_normal_list[i] = Eigen::Vector3f::Zero();
	}
	for (int idx = 0; idx < idxsgroup.size(); idx++) {//Add O(m)
		const int & iface = idxsgroup[idx];
		obj_face & f = face_list[iface];
		if (f.vertex_count == 3) {
			local_vertex_normal_list[f.vertex_index[0]] += face_normal_list[2 * iface];
			local_vertex_normal_list[f.vertex_index[1]] += face_normal_list[2 * iface];
			local_vertex_normal_list[f.vertex_index[2]] += face_normal_list[2 * iface];
		}
		else if (f.vertex_count == 4) {
			local_vertex_normal_list[f.vertex_index[0]] += face_normal_list[2 * iface + 1];
			local_vertex_normal_list[f.vertex_index[1]] += face_normal_list[2 * iface + 1] + face_normal_list[2 * iface];
			local_vertex_normal_list[f.vertex_index[2]] += face_normal_list[2 * iface];
			local_vertex_normal_list[f.vertex_index[3]] += face_normal_list[2 * iface + 1] + face_normal_list[2 * iface];
		}
	}
	int count = 0;
	int last_ = vertex_normal_list.size();
	for (int i = 0; i < nV; i++) {//Set Index & Normalize O(N)
		if (scoreboard[i] == 0) {
			scoreboard[i] = count++;
			if (local_vertex_normal_list[i].norm() != 0.f)
				local_vertex_normal_list[i].normalize();
			vertex_normal_list.push_back(local_vertex_normal_list[i]);
		}
	}
	for (int idx = 0; idx < idxsgroup.size(); idx++) {//O(m) 
		const int & iface = idxsgroup[idx];
		obj_face & f = face_list[iface];
		for (int j = 0; j < f.vertex_count; j++) {
			if (forcefill || f.normal_index[j] == -1) {
				f.normal_index[j] = last_ + scoreboard[f.vertex_index[j]];
			}
		}
	}
}

void SMesh::smoothAlgorithmII(const std::vector<int> & idxsgroup, bool forcefill)
{
	updateFaceNormals();

	std::map<int, int> amap;
	for (int idx = 0; idx < idxsgroup.size(); idx++) {// O(mlogm)
		const int & iface = idxsgroup[idx];
		obj_face &f = face_list[iface];
		for (int j = 0; j < f.vertex_count; j++) {
			amap[f.vertex_index[j]] = 0;
		}
	}
	int count = 0;
	for (std::map<int, int>::iterator it = amap.begin(); it != amap.end(); it++) { // O(m)
		it->second = count++;
	}
	std::vector<Eigen::Vector3f> local_vertex_normal_list(count, Eigen::Vector3f::Zero());// O(m)
	for (int idx = 0; idx < idxsgroup.size(); idx++) {// O(mlogm)
		const int & iface = idxsgroup[idx];
		obj_face & f = face_list[iface];
		if (f.vertex_count == 3) {
			local_vertex_normal_list[amap[f.vertex_index[0]]] += face_normal_list[2 * iface];
			local_vertex_normal_list[amap[f.vertex_index[1]]] += face_normal_list[2 * iface];
			local_vertex_normal_list[amap[f.vertex_index[2]]] += face_normal_list[2 * iface];
		}
		else if (f.vertex_count == 4) {
			local_vertex_normal_list[amap[f.vertex_index[0]]] += face_normal_list[2 * iface + 1];
			local_vertex_normal_list[amap[f.vertex_index[1]]] += face_normal_list[2 * iface + 1] + face_normal_list[2 * iface];
			local_vertex_normal_list[amap[f.vertex_index[2]]] += face_normal_list[2 * iface];
			local_vertex_normal_list[amap[f.vertex_index[3]]] += face_normal_list[2 * iface + 1] + face_normal_list[2 * iface];
		}
	}
	int last_ = vertex_normal_list.size();
	for (int i = 0; i < count; i++) {// Normalize O(m)
		if (local_vertex_normal_list[i].norm() != 0.f)
			local_vertex_normal_list[i].normalize();
		vertex_normal_list.push_back(local_vertex_normal_list[i]);
	}
	for (int idx = 0; idx < idxsgroup.size(); idx++) {// O(mlogm)
		const int & iface = idxsgroup[idx];
		obj_face & f = face_list[iface];
		for (int j = 0; j < f.vertex_count; j++) {
			if (forcefill || f.normal_index[j] == -1) {
				f.normal_index[j] = last_ + amap[f.vertex_index[j]];
			}
		}
	}
}

void SMesh::roughAlgorithm(const std::vector<int> & idxsgroup, bool forcefill)
{
	int last_ = vertex_normal_list.size();
	for (int idx = 0; idx < idxsgroup.size(); idx++) {
		const int & iface = idxsgroup[idx];
		obj_face & f = face_list[iface];
		if (f.vertex_count == 3) // triangle face
			vertex_normal_list.push_back(face_normal_list[2 * iface]);
		else if (f.vertex_count == 4) { // mix two triangle faces
			Eigen::Vector3f tmp = face_normal_list[2 * iface] + face_normal_list[2 * iface + 1];
			if (tmp.norm() != 0.f)
				tmp.normalize();
			vertex_normal_list.push_back(tmp);
		}
		for (int j = 0; j < f.vertex_count; j++) {
			if (forcefill || f.normal_index[j] == -1)
				f.normal_index[j] = last_ + idx;
		}
	}
}

void SMesh::updateNormalsCompulsory()
{
	updateFaceNormals();
	vertex_normal_list.clear();//re-generate
	for (int iList = 0; iList < group_list.size(); iList++)
	{
		obj_group & group = group_list[iList];
		int begin, end;
		int smooth_group;
		std::map<int, std::vector<int>> s_group;//sum+range
		std::map<int, std::vector<int>>::iterator it;
		const int nSeg = group.smooth_info.size();
		if (nSeg) {
			begin = group.smooth_info[0].Index();
			for (int iseg = 1; iseg <= nSeg; iseg++) { // ALL SMOOTH GROUP
				smooth_group = group.smooth_info[iseg - 1].SmoothGroup();
				if (iseg == nSeg)
					end = group.group_info.End();
				else
					end = group.smooth_info[iseg].Index();
				it = s_group.find(smooth_group);
				if (it == s_group.end()) {
					s_group.insert(std::pair<int, std::vector<int>>(smooth_group, std::vector<int>()));
					it = s_group.find(smooth_group);
				}
				for (int i = begin; i < end; i++)
					it->second.push_back(i);
				begin = end;
			}
		}
		for (it = s_group.begin(); it != s_group.end(); it++) {
			std::vector<int> & idxsgroup = it->second;
			smooth_group = it->first;
			if (smooth_group) { //generate anyway
				if (policyMaking(idxsgroup.size(), vertex_list.size()))
					smoothAlgorithmI(idxsgroup, true);// O(m/N)
				else
					smoothAlgorithmII(idxsgroup, true);// O(mlogm)
			}
			else {
				roughAlgorithm(idxsgroup, true);// O(m)
			}
		}
	}
}


void SMesh::updateNormals()
{
	updateFaceNormals();
	for (int iList = 0; iList < group_list.size(); iList++)
	{
		obj_group & group = group_list[iList];
		int begin, end;
		int smooth_group;
		std::map<int, std::vector<int>> s_group;//sum+range
		std::map<int, std::vector<int>>::iterator it;
		const int nSeg = group.smooth_info.size();
		if (nSeg) {
			begin = group.smooth_info[0].Index();
			for (int iseg = 1; iseg <= nSeg; iseg++) { // ALL SMOOTH GROUP
				smooth_group = group.smooth_info[iseg - 1].SmoothGroup();
				if (iseg == nSeg)
					end = group.group_info.End();
				else
					end = group.smooth_info[iseg].Index();
				it = s_group.find(smooth_group);
				if (it == s_group.end()) {
					s_group.insert(std::pair<int, std::vector<int>>(smooth_group, std::vector<int>()));
					it = s_group.find(smooth_group);
				}
				for (int i = begin; i < end; i++)
					it->second.push_back(i);
				begin = end;
			}
		}
		for (it = s_group.begin(); it != s_group.end(); it++) {
			std::vector<int> & idxsgroup = it->second;
			smooth_group = it->first;
			if (needGenNormals(idxsgroup)) {// Gen normals
				if (smooth_group) {
					if (policyMaking(idxsgroup.size(), vertex_list.size()))
						smoothAlgorithmI(idxsgroup, false);// O(m/N)
					else
						smoothAlgorithmII(idxsgroup, false);// O(mlogm)
				}
				else {
					roughAlgorithm(idxsgroup, false);// O(m)
				}
			}
		}
	}
}

/*
extract object
if set false, copy each segment seperately, may cause redundancy
if set true, try to compress data, time consuming
*/
bool SMesh::extractObject(SMesh & submesh, const char * filename, bool checkvertex, bool checknormal, bool checktexture)
{
	std::map<std::string, std::vector<int>>::iterator it = group_map.find(std::string(filename));
	if (it == group_map.end())
		return false;
	// Algorithm I
	std::vector<int> scoreboard;
	// Algorithm II
	std::map<int, int> amap;
	//
	submesh.clear();
	submesh._name = this->_name;
	strncpy(submesh.scene_filename, this->scene_filename, OBJ_FILENAME_LENGTH);
	strncpy(submesh.material_filename, this->material_filename, OBJ_FILENAME_LENGTH);
	std::vector<int> & group_list_ids = it->second;
	int count = 0;
	for (int iSeg = 0; iSeg < group_list_ids.size(); iSeg++) {
		count += group_list[group_list_ids[iSeg]].group_info.Offset();
	}
	int iSub = 0;
	//////////////// Gen Group List
	scoreboard.resize(group_list_ids.size());
	for (int iSeg = 0; iSeg < group_list_ids.size(); iSeg++) {
		obj_group & objg_src = this->group_list[group_list_ids[iSeg]];
		submesh.group_list.push_back(objg_src);
		obj_group & objg_tar = submesh.group_list[iSeg];
		objg_tar.group_info.SetBegin(iSub);
		objg_tar.group_info.SetEnd(objg_src.group_info.Offset() + iSub);
		for (int j = 0; j < objg_src.smooth_info.size(); j++) {
			objg_tar.smooth_info[j].SetIndex(iSub + (objg_src.smooth_info[j].Index() - objg_src.group_info.Begin()));
		}
		iSub += objg_src.group_info.Offset();
		scoreboard[iSeg] = iSeg;
	}
	submesh.group_map[it->first] = scoreboard;
	//////////////// Gen Face List, Gen Material List
	iSub = 0;
	submesh.face_list.resize(count);
	scoreboard.resize(material_list.size());
	for (int i = 0; i < material_list.size(); i++)
		scoreboard[i] = -1;
	for (int iSeg = 0; iSeg < group_list_ids.size(); iSeg++) {
		int iList = group_list_ids[iSeg];
		int begin = group_list[iList].group_info.Begin();
		int end = group_list[iList].group_info.End();
		for (int iface = begin; iface < end; iface++) {
			const obj_face & f = face_list[iface];
			submesh.face_list[iSub++] = f;//copy face data
			if (f.material_index >= 0) {// -1
				scoreboard[f.material_index] = 0;
			}
		}
	}
	iSub = 0;
	for (int i = 0; i< material_list.size(); i++)// gen material list
		if (scoreboard[i] == 0) {
			scoreboard[i] = iSub++;
			submesh.material_list.push_back(material_list[i]);
		}
	for (int i = 0; i < count; i++) {
		obj_face & f = submesh.face_list[i];
		if (f.material_index >= 0) //modify material index
			f.material_index = scoreboard[f.material_index];
	}
	printf("INFO: extract %lu faces\n", submesh.face_list.size());
	//////////////// Gen Vertex List
	if (!checkvertex) {
		iSub = 0;
		for (int iSeg = 0; iSeg < group_list_ids.size(); iSeg++) {
			int iList = group_list_ids[iSeg];
			int begin = group_list[iList].group_info.Begin();
			int end = group_list[iList].group_info.End();
			int vmin = 0x7fffffff;
			int vmax = 0x80000000;
			for (int iface = begin; iface < end; iface++) {
				const obj_face & f = face_list[iface];
				for (int j = 0; j < f.vertex_count; j++) {
					if (f.vertex_index[j] >= 0 && f.vertex_index[j] > vmax)
						vmax = f.vertex_index[j];
					if (f.vertex_index[j] >= 0 && f.vertex_index[j] < vmin)
						vmin = f.vertex_index[j];
				}
			} vmax++;
			for (int i = vmin; i < vmax; i++)
				submesh.vertex_list.push_back(this->vertex_list[i]);
			for (int iface = iSub; iface < iSub + (end - begin); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j<f.vertex_count; j++)
					if (f.vertex_index[j] >= 0)
						f.vertex_index[j] = iSub + (f.vertex_index[j] - vmin);
			}
			iSub += (end - begin);
		}
	}
	else {
		if (policyMaking(count, vertex_list.size())) {// ALGORITHM I
			const int vn = vertex_list.size();
			scoreboard.resize(vn);
			for (int i = 0; i < vn; i++)
				scoreboard[i] = -1;
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.vertex_index[j] >= 0)
						scoreboard[f.vertex_index[j]] = 0;
			}
			iSub = 0;
			for (int i = 0; i < vn; i++) // gen vertex list
				if (scoreboard[i] == 0) {
					scoreboard[i] = iSub++;
					submesh.vertex_list.push_back(this->vertex_list[i]);
				}
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.vertex_index[j] >= 0)
						f.vertex_index[j] = scoreboard[f.vertex_index[j]];
			}
		}
		else { //ALGORITHM II
			amap.clear();
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {// O(mlogm)
				obj_face &f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.vertex_index[j] >= 0)
						amap[f.vertex_index[j]] = 0;
			}
			iSub = 0;
			for (std::map<int, int>::iterator it = amap.begin(); it != amap.end(); it++) { // O(m)
				it->second = iSub++;
				submesh.vertex_list.push_back(this->vertex_list[it->first]);// gen vertex list
			}
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.vertex_index[j] >= 0)
						f.vertex_index[j] = amap[f.vertex_index[j]];
			}
		}
	}
	printf("INFO: extract %lu vertexes\n", submesh.vertex_list.size());
	//////////////// Gen Normal List
	if (!checknormal) {
		iSub = 0;
		for (int iSeg = 0; iSeg < group_list_ids.size(); iSeg++) {
			int iList = group_list_ids[iSeg];
			int begin = group_list[iList].group_info.Begin();
			int end = group_list[iList].group_info.End();
			int vmin = 0x7fffffff;
			int vmax = 0x80000000;
			for (int iface = begin; iface < end; iface++) {
				const obj_face & f = face_list[iface];
				for (int j = 0; j < f.vertex_count; j++) {
					if (f.normal_index[j] >= 0 && f.normal_index[j] > vmax)
						vmax = f.normal_index[j];
					if (f.normal_index[j] >= 0 && f.normal_index[j] < vmin)
						vmin = f.normal_index[j];
				}
			} vmax++;
			for (int i = vmin; i < vmax; i++)
				submesh.vertex_normal_list.push_back(this->vertex_normal_list[i]);
			for (int iface = iSub; iface < iSub + (end - begin); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j<f.vertex_count; j++)
					if (f.normal_index[j] >= 0)
						f.normal_index[j] = iSub + (f.normal_index[j] - vmin);
			}
			iSub += (end - begin);
		}
	}
	else {
		if (policyMaking(count, vertex_normal_list.size())) {// ALGORITHM I
			const int vn = vertex_normal_list.size();
			scoreboard.resize(vn);
			for (int i = 0; i < vn; i++)
				scoreboard[i] = -1;
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.normal_index[j] >= 0)
						scoreboard[f.normal_index[j]] = 0;
			}
			iSub = 0;
			for (int i = 0; i < vn; i++) // gen normal list
				if (scoreboard[i] == 0) {
					scoreboard[i] = iSub++;
					submesh.vertex_normal_list.push_back(this->vertex_normal_list[i]);
				}
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.normal_index[j] >= 0)
						f.normal_index[j] = scoreboard[f.normal_index[j]];
			}
		}
		else { //ALGORITHM II
			amap.clear();
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {// O(mlogm)
				obj_face &f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.normal_index[j] >= 0)
						amap[f.normal_index[j]] = 0;
			}
			iSub = 0;
			for (std::map<int, int>::iterator it = amap.begin(); it != amap.end(); it++) { // O(m)
				it->second = iSub++;
				submesh.vertex_normal_list.push_back(this->vertex_normal_list[it->first]);// gen normal list
			}
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.normal_index[j] >= 0)
						f.normal_index[j] = amap[f.normal_index[j]];
			}
		}
	}
	printf("INFO: extract %lu normals\n", submesh.vertex_normal_list.size());
	//////////////// Gen Texcoord List
	if (!checktexture) {
		iSub = 0;
		for (int iSeg = 0; iSeg < group_list_ids.size(); iSeg++) {
			int iList = group_list_ids[iSeg];
			int begin = group_list[iList].group_info.Begin();
			int end = group_list[iList].group_info.End();
			int vmin = 0x7fffffff;
			int vmax = 0x80000000;
			for (int iface = begin; iface < end; iface++) {
				const obj_face & f = face_list[iface];
				for (int j = 0; j < f.vertex_count; j++) {
					if (f.texture_index[j] >= 0 && f.texture_index[j] > vmax)
						vmax = f.texture_index[j];
					if (f.texture_index[j] >= 0 && f.texture_index[j] < vmin)
						vmin = f.texture_index[j];
				}
			} vmax++;
			for (int i = vmin; i < vmax; i++)
				submesh.vertex_texture_list.push_back(this->vertex_texture_list[i]);
			for (int iface = iSub; iface < iSub + (end - begin); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j<f.vertex_count; j++)
					if (f.texture_index[j] >= 0)
						f.texture_index[j] = iSub + (f.texture_index[j] - vmin);
			}
			iSub += (end - begin);
		}
	}
	else {
		if (policyMaking(count, vertex_texture_list.size())) {// ALGORITHM I
			const int vn = vertex_texture_list.size();
			scoreboard.resize(vn);
			for (int i = 0; i < vn; i++)
				scoreboard[i] = -1;
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.texture_index[j] >= 0)
						scoreboard[f.texture_index[j]] = 0;
			}
			iSub = 0;
			for (int i = 0; i < vn; i++) // gen texture list
				if (scoreboard[i] == 0) {
					scoreboard[i] = iSub++;
					submesh.vertex_texture_list.push_back(this->vertex_texture_list[i]);
				}
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.texture_index[j] >= 0)
						f.texture_index[j] = scoreboard[f.texture_index[j]];
			}
		}
		else { //ALGORITHM II
			amap.clear();
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {// O(mlogm)
				obj_face &f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.texture_index[j] >= 0)
						amap[f.texture_index[j]] = 0;
			}
			iSub = 0;
			for (std::map<int, int>::iterator it = amap.begin(); it != amap.end(); it++) { // O(m)
				it->second = iSub++;
				submesh.vertex_texture_list.push_back(this->vertex_texture_list[it->first]);// gen texture list
			}
			for (int iface = 0; iface < submesh.face_list.size(); iface++) {
				obj_face & f = submesh.face_list[iface];
				for (int j = 0; j < f.vertex_count; j++)
					if (f.texture_index[j] >= 0)
						f.texture_index[j] = amap[f.texture_index[j]];
			}
		}
	}
	printf("INFO: extract %lu texture UVs\n", submesh.vertex_texture_list.size());
	return true;
}


void SMesh::renderShader(
	int showType, const char* filename,
	int sel_material, int normal, int texture
) {
	if (vertex_list.size() == 0)
		return;
	if (face_list.size() == 0)
		return;

	std::map<std::string, std::vector<int>>::iterator it;
	std::vector<int> group_list_index;
	if (filename != NULL) {
		it = group_map.find(std::string(filename));
		if (it == group_map.end())
			return;
		group_list_index = it->second;
		if (group_list_index.empty()) //default may contains nothing
			return;
	}

	//// == renderFaces
	{
		//Suppose Filename Is Valid
		obj_render* render;
		std::map<std::string, obj_render>::iterator it;
		if (filename != NULL) {
			std::string sfn(filename);
			it = sub_render.find(sfn);
			if (it == sub_render.end()) {
				sub_render.insert(std::pair<std::string, obj_render>(sfn, obj_render()));
				it = sub_render.find(sfn);
			}
			render = &(it->second);
		}
		else
			render = &gathering_render;

		genFastView(group_list_index, render);//

		int nS = material_list.size();
		if (nS == 0) nS = 1;

		for (int i = 0; i < nS; i++)
			//for(int i = 0; i<4;i++)
		{
			if (render->_fast_view_verts[i].size() == 0)
				continue;
			
			// negative sel_material means render all 
			if (sel_material >= 0 && i != sel_material)
				continue;
			
			//int& iS = i;
			
			std::vector<int> attrib_idx;
			glEnableVertexAttribArray(0); // Vertex
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (float *)render->_fast_view_verts[i].data());
			if (normal != -1 ) { // Normal
				glEnableVertexAttribArray(normal);
				if (showType & SW_SMOOTH)
					glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 0, (float*)render->_fast_view_vertex_normals[i].data());
				else
					glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 0, (float*)render->_fast_view_face_normals[i].data());				
				attrib_idx.push_back(normal);
			}
			if (texture != -1) { // TexCoord
				glEnableVertexAttribArray(texture);
				glVertexAttribPointer(texture, 2, GL_FLOAT, GL_FALSE, 0, (float*)render->_fast_view_texcoords[i].data());
				attrib_idx.push_back(texture);
			}
			glDrawArrays(GL_TRIANGLES, 0, render->_fast_view_verts[i].size());
			for (size_t i = 0; i < attrib_idx.size(); i++) 
				glDisableVertexAttribArray(attrib_idx[i]);
			glDisableVertexAttribArray(0);
		}
	}
}

void SMesh::render(int showType, const char * filename)
{
	if (vertex_list.size() == 0)
		return;
	if (face_list.size() == 0)
		return;

	std::map<std::string, std::vector<int>>::iterator it;
	std::vector<int> group_list_index;
	if (filename != NULL) {
		it = group_map.find(std::string(filename));
		if (it == group_map.end())
			return;
		group_list_index = it->second;
		if (group_list_index.empty()) //default may contains nothing
			return;
	}

	const Eigen::Vector3f * vertices = &vertex_list[0];
	const obj_face * faces = &face_list[0];
	const obj_material * mats = 0;
	int nfaces = face_list.size();
	int nverts = vertex_list.size();
	if (material_list.size() > 0)
		mats = &material_list[0];

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if (showType & SW_LIGHTING)
		glEnable(GL_LIGHTING);
	else
		glDisable(GL_LIGHTING);

	if (showType & SW_E) {
		glEnable(GL_POLYGON_OFFSET_FILL);//set polygon offset
		glPolygonOffset(1.f, 1.f);
	}

	if (showType & SW_F)
		renderFaces(group_list_index, showType, filename);

	if (showType & SW_E) {
		glDisable(GL_LIGHTING);
		glLineWidth(1.f);
		glBegin(GL_LINES);
		if (filename == NULL) {//render all
			for (int i = 0; i < nfaces; i++) {
				const int * vidx = faces[i].vertex_index;
				int nfc = faces[i].vertex_count;
				for (int j = 0; j < nfc; j++) {
					glColor3f(0.2f, 0.3f, 0.6f);
					glVertex3fv((const float *)&vertices[vidx[j]]);
					glColor3f(0.2f, 0.3f, 0.6f);
					glVertex3fv((const float *)&vertices[vidx[(j + 1) % nfc]]);
				}
			}
		}
		else {
			for (int igroup = 0; igroup < group_list_index.size(); igroup++)
			{
				obj_group & render = group_list[group_list_index[igroup]];
				for (int i = render.group_info.Begin(); i < render.group_info.End(); i++) {
					const int * vidx = faces[i].vertex_index;
					int nfc = faces[i].vertex_count;
					for (int j = 0; j < nfc; j++) {
						glColor3f(0.2f, 0.3f, 0.6f);
						glVertex3fv((const float *)&vertices[vidx[j]]);
						glColor3f(0.2f, 0.3f, 0.6f);
						glVertex3fv((const float *)&vertices[vidx[(j + 1) % nfc]]);
					}
				}
			}
		}
		glEnd();
		glColor3f(1.f, 1.f, 1.f);
	}

	if (showType & SW_V)
	{
		glDisable(GL_LIGHTING);
		glPointSize(5.f);
		glBegin(GL_POINTS);
		if (filename == NULL) {
			for (int i = 0; i < nverts; i++)
			{
				glColor3f(0.2f, 0.3f, 0.6f);
				glVertex3fv(vertices[i].data());
			}
		}
		else {
			for (int igroup = 0; igroup < group_list_index.size(); igroup++)
			{
				obj_group & render = group_list[group_list_index[igroup]];
				for (int i = render.group_info.Begin(); i < render.group_info.End(); i++) {
					const int * vidx = faces[i].vertex_index;
					int nfc = faces[i].vertex_count;
					for (int j = 0; j < nfc; j++) {
						glColor3f(0.2f, 0.3f, 0.6f);
						glVertex3fv(vertices[i].data());
					}
				}
			}
		}
		glEnd();

		glColor3f(1.f, 1.f, 1.f);
		glPointSize(1.f);
	}

	glDisable(GL_POLYGON_OFFSET_FILL);

	glPopAttrib();
}

void SMesh::genFastView(const std::vector<int> & group_list_index, obj_render * render) {

	if (render->_valid_fast_view) return;
	int list_size = material_list.size();
	if (list_size == 0) list_size = 1;
	updateFaceNormals();
	// TRIANGLES | QUADS || TRIANGLES | QUADS 
	render->_fast_view_verts.clear();
	render->_fast_view_vertex_normals.clear();
	render->_fast_view_face_normals.clear();
	render->_fast_view_texcoords.clear();
	//
	render->_fast_view_verts.resize(list_size);
	render->_fast_view_vertex_normals.resize(list_size);
	render->_fast_view_face_normals.resize(list_size);
	render->_fast_view_texcoords.resize(list_size);
	//
	if (group_list_index.size() == 0) { //Render all
		for (int iface = 0; iface < face_list.size(); iface++)
		{
			const obj_face & f = face_list[iface];
			int iS = 0;
			if (material_list.size() != 0 && f.material_index >= 0
				&& f.material_index < material_list.size())
				iS = f.material_index;

			std::vector<Eigen::Vector3f>& verts = render->_fast_view_verts[iS];
			std::vector<Eigen::Vector3f>& vertex_normals = render->_fast_view_vertex_normals[iS];
			std::vector<Eigen::Vector3f>& face_normals = render->_fast_view_face_normals[iS];
			std::vector<Eigen::Vector2f>& tex_coords = render->_fast_view_texcoords[iS];

			if (f.vertex_count == 3) {
				verts.push_back(vertex_list[f.vertex_index[0]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[0]]); face_normals.push_back(face_normal_list[2 * iface]);
				if (f.texture_index[0] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[0]]);
				verts.push_back(vertex_list[f.vertex_index[1]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[1]]); face_normals.push_back(face_normal_list[2 * iface]);
				if (f.texture_index[1] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[1]]);
				verts.push_back(vertex_list[f.vertex_index[2]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[2]]); face_normals.push_back(face_normal_list[2 * iface]);
				if (f.texture_index[2] != -1)  tex_coords.push_back(vertex_texture_list[f.texture_index[2]]);

			}
			else if (f.vertex_count == 4) {
				// 0 1 2
				verts.push_back(vertex_list[f.vertex_index[0]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[0]]); face_normals.push_back(face_normal_list[2 * iface]);
				if (f.texture_index[0] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[0]]);
				verts.push_back(vertex_list[f.vertex_index[1]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[1]]); face_normals.push_back(face_normal_list[2 * iface]);
				if (f.texture_index[1] != -1)  tex_coords.push_back(vertex_texture_list[f.texture_index[1]]);
				verts.push_back(vertex_list[f.vertex_index[2]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[2]]); face_normals.push_back(face_normal_list[2 * iface]);
				if (f.texture_index[2] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[2]]);
				// 0 2 3
				verts.push_back(vertex_list[f.vertex_index[0]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[0]]); face_normals.push_back(face_normal_list[2 * iface + 1]);
				if (f.texture_index[0] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[0]]);
				verts.push_back(vertex_list[f.vertex_index[2]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[2]]); face_normals.push_back(face_normal_list[2 * iface + 1]);
				if (f.texture_index[2] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[2]]);
				verts.push_back(vertex_list[f.vertex_index[3]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[3]]); face_normals.push_back(face_normal_list[2 * iface + 1]);
				if (f.texture_index[3] != -1)	tex_coords.push_back(vertex_texture_list[f.texture_index[3]]);
			}
			else {
				printf("error\n");
			}
		}
	}
	else { // Render part
		for (int iList = 0; iList < group_list_index.size(); iList++) {
			obj_group & g = group_list[group_list_index[iList]];
			for (int iface = g.group_info.Begin(); iface < g.group_info.End(); iface++) {
				const obj_face & f = face_list[iface];
				int iS = 0;
				if (material_list.size() != 0 && f.material_index >= 0
					&& f.material_index < material_list.size())
					iS = f.material_index;

				std::vector<Eigen::Vector3f>& verts = render->_fast_view_verts[iS];
				std::vector<Eigen::Vector3f>& vertex_normals = render->_fast_view_vertex_normals[iS];
				std::vector<Eigen::Vector3f>& face_normals = render->_fast_view_face_normals[iS];
				std::vector<Eigen::Vector2f>& tex_coords = render->_fast_view_texcoords[iS];

				if (f.vertex_count == 3) {
					verts.push_back(vertex_list[f.vertex_index[0]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[0]]); face_normals.push_back(face_normal_list[2 * iface]);
					if (f.texture_index[0] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[0]]);
					verts.push_back(vertex_list[f.vertex_index[1]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[1]]); face_normals.push_back(face_normal_list[2 * iface]);
					if (f.texture_index[1] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[1]]);
					verts.push_back(vertex_list[f.vertex_index[2]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[2]]); face_normals.push_back(face_normal_list[2 * iface]);
					if (f.texture_index[2] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[2]]);
				}
				else if (f.vertex_count == 4) {
					// 1 2 3
					verts.push_back(vertex_list[f.vertex_index[1]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[1]]); face_normals.push_back(face_normal_list[2 * iface]);
					if (f.texture_index[1] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[1]]);
					verts.push_back(vertex_list[f.vertex_index[2]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[2]]); face_normals.push_back(face_normal_list[2 * iface]);
					if (f.texture_index[2] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[2]]);
					verts.push_back(vertex_list[f.vertex_index[3]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[3]]); face_normals.push_back(face_normal_list[2 * iface]);
					if (f.texture_index[3] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[3]]);
					// 0 1 3
					verts.push_back(vertex_list[f.vertex_index[0]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[0]]); face_normals.push_back(face_normal_list[2 * iface + 1]);
					if (f.texture_index[0] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[0]]);
					verts.push_back(vertex_list[f.vertex_index[1]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[1]]); face_normals.push_back(face_normal_list[2 * iface + 1]);
					if (f.texture_index[1] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[1]]);
					verts.push_back(vertex_list[f.vertex_index[3]]); vertex_normals.push_back(vertex_normal_list[f.normal_index[3]]); face_normals.push_back(face_normal_list[2 * iface + 1]);
					if (f.texture_index[3] != -1) tex_coords.push_back(vertex_texture_list[f.texture_index[3]]);
				}
				else {
					printf("error!\n");
				}
			}
		}
	}
	render->_valid_fast_view = true;
}

void SMesh::renderFaces(const std::vector<int> & group_list_index, int showType, const char * filename)
{
	//Suppose Filename Is Valid
	obj_render * render;
	std::map<std::string, obj_render>::iterator it;
	if (filename != NULL) {
		std::string sfn(filename);
		it = sub_render.find(sfn);
		if (it == sub_render.end()) {
			sub_render.insert(std::pair<std::string, obj_render>(sfn, obj_render()));
			it = sub_render.find(sfn);
		}
		render = &(it->second);
	}
	else
		render = &gathering_render;

	genFastView(group_list_index, render);//

	bool enableTexture = ((showType & SW_TEXTURE) && (vertex_texture_list.size() > 0));
	//bool enableTexture = showType & SW_TEXTURE;
	bool subEnableTexture;

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	//if (enableTexture)
	//	glEnableClientState(GL_TEXTURE_COORD_ARRAY);

	int nS = material_list.size();
	if (nS == 0) nS = 1;

	//float debug_vertex[] = {-1.f,0.f,0.f, 1.f,0.f,0.f,  0.f,1.f,0.f};
	//float debug_normal[] = { 0.f,0.f,1.f, 0.f,0.f,1.f, 0.f,0.f,1.f };
	//glVertexPointer(3, GL_FLOAT, 0, debug_vertex);
	//glNormalPointer(GL_FLOAT, 0, debug_normal);
	//glDrawArrays(GL_TRIANGLES, 0, 3);

	

	for (int i = 0; i < nS; i++)
	//for(int i = 0; i<4;i++)
	{
		if (render->_fast_view_verts[i].size() == 0)
			continue;
		//
		int & iS = i;

		subEnableTexture = enableTexture && render->_fast_view_texcoords[i].size() > 0 ;

		//
		if (material_list.size() == 0)
			default_material.drawMat(subEnableTexture);
		else
			material_list[iS].drawMat(subEnableTexture);
		glVertexPointer(3, GL_FLOAT, 0, render->_fast_view_verts[i].data());
		if (showType & SW_SMOOTH)
			glNormalPointer(GL_FLOAT, 0, render->_fast_view_vertex_normals[i].data());
		else
			glNormalPointer(GL_FLOAT, 0, render->_fast_view_face_normals[i].data());
		//if (enableTexture && render->_fast_view_texcoords[i].size()) {
		if(subEnableTexture){
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			glTexCoordPointer(2, GL_FLOAT, 0, render->_fast_view_texcoords[i].data());
		}
		//
		glDrawArrays(GL_TRIANGLES, 0, render->_fast_view_verts[i].size());
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		
	}
	

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	glDisable(GL_TEXTURE_2D); // otherwise, may influence edge&point rendering ? 
}

SMesh::obj_boundingbox SMesh::boundingBox(const char * filename)
{
	SMesh::obj_boundingbox bb;
	if (filename == NULL) {
		for (int i = 0; i < vertex_list.size(); i++) {
			for (int j = 0; j < 3; j++) {
				if (vertex_list[i][j] < bb.minmax[0][j])
					bb.minmax[0][j] = vertex_list[i][j];
				if (vertex_list[i][j] > bb.minmax[1][j])
					bb.minmax[1][j] = vertex_list[i][j];
			}
		}
	}
	else {
		std::map<std::string, std::vector<int>>::iterator it = group_map.find(std::string(filename));
		if (it != group_map.end()) {
			for (int iseg = 0; iseg < it->second.size(); iseg++) {
				const int & list_id = it->second[iseg];
				obj_group & objg = group_list[list_id];
				for (int iface = objg.group_info.Begin(); iface < objg.group_info.End(); iface++) {
					obj_face & f = face_list[iface];
					for (int iv = 0; iv < f.vertex_count; iv++) {
						Eigen::Vector3f & vec = vertex_list[f.vertex_index[iv]];
						for (int j = 0; j < 3; j++) {
							if (vec[j] < bb.minmax[0][j])
								bb.minmax[0][j] = vec[j];
							if (vec[j] > bb.minmax[1][j])
								bb.minmax[1][j] = vec[j];
						}
					}
				}
			}
		}
	}
	return bb;
}

void SMesh::deBias(const obj_boundingbox * bb)
{
	obj_boundingbox ibb;
	if (bb == NULL) {
		ibb = boundingBox();
		bb = &ibb;
	}
	Eigen::Vector3f Offset;
	Offset = (bb->lowerBound() + bb->upperBound()) / 2.f;
	for (int i = 0; i < vertex_list.size(); i++) {
		vertex_list[i] -= Offset;
	}
}

void SMesh::afterPoseChanged(bool update_normal) {
	// re-compute normals
	if (update_normal) {
		_valid_face_normal_list = false;
		updateNormalsCompulsory();
	}
	// clear render buffer
	gathering_render._valid_fast_view = false;
	for (std::map<std::string, obj_render>::iterator it = sub_render.begin(); it != sub_render.end(); it++) {
		it->second._valid_fast_view = false;
	}
}

// this is not safe though
// only for debugging
void SMesh::scale(float v) {
	for (size_t i = 0; i < vertex_list.size(); i++) {
		vertex_list[i] *= v;
	}
}

void SMesh::offset(float * v) {
	Eigen::Vector3f offset(v[0], v[1], v[2]);
	for (size_t i = 0; i < vertex_list.size(); i++)
		vertex_list[i] += offset;
}

////////////////////////
//
SMesh::obj_boundingbox::obj_boundingbox() {
	minmax[0] << FLT_MAX, FLT_MAX, FLT_MAX;
	minmax[1] << -FLT_MAX, -FLT_MAX, -FLT_MAX;
}

Eigen::Vector3f SMesh::obj_boundingbox::lowerBound() const { return minmax[0]; }
Eigen::Vector3f SMesh::obj_boundingbox::upperBound() const { return minmax[1]; }

////////////////////////
//
SMesh::obj_pair::obj_pair(int _first, int _second) { first = _first; second = _second; }
SMesh::obj_pair::obj_pair() { first = 0; second = 0; }
// group info
int SMesh::obj_pair::Begin() const { return first; }
int SMesh::obj_pair::End() const { return second; }
int SMesh::obj_pair::Offset() const { return second - first; }
void SMesh::obj_pair::SetBegin(int a) { first = a; }
void SMesh::obj_pair::SetEnd(int a) { second = a; }
void SMesh::obj_pair::SetOffset(int a) { second = first + a; }
// smooth group info
int SMesh::obj_pair::SmoothGroup() const { return first; }
int SMesh::obj_pair::Index() const { return second; }
void SMesh::obj_pair::SetSmoothGroup(int a) { first = a; }
void SMesh::obj_pair::SetIndex(int a) { second = a; }

////////////////////////
//
void SMesh::obj_group::clear()
{
	smooth_info.clear();
}

////////////////////////
//
SMesh::obj_render::obj_render()
{
	_valid_fast_view = false;
}

void SMesh::obj_render::clear()
{
	_fast_view_verts.clear();
	_fast_view_vertex_normals.clear();
	_fast_view_face_normals.clear();
	_fast_view_texcoords.clear();
	_valid_fast_view = false;
}

////////////////////////
//
SMesh::obj_material::obj_material()
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
	trans = 0.f;// Tr
	glossy = 98;
	shiny = 0;
	refract_index = 1;
	illum = -1;
}

void SMesh::obj_material::drawMat(int isTextureEnabled) const
{
	float color[4];

	color[0] = amb[0];
	color[1] = amb[1];
	color[2] = amb[2];
	color[3] = 1 - trans;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);

	color[0] = (float)diff[0];
	color[1] = (float)diff[1];
	color[2] = (float)diff[2];
	color[3] = (float)(1-trans);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);

	color[0] = (float)spec[0];
	color[1] = (float)spec[1];
	color[2] = (float)spec[2];
	color[3] = (float)(1-trans);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);

	//cos(theta)^shiny
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, (float)(shiny));

	//draw texture
	if (isTextureEnabled && texture.texture_id > 0)
	{
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, texture.texture_id);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.image.width(),
			texture.image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, texture.image.data());
	}
	else {
		glDisable(GL_TEXTURE_2D);
	}
}

void SMesh::obj_material::generateTextures()
{
	if (texture.texture_id == 0)
		return;
	glGenTextures(1, &texture.texture_id);

	if (texture.texture_id == 0)
		printf("ERROR: Get OpenGL contex failed!\n");

	glBindTexture(GL_TEXTURE_2D, texture.texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}


/////////////////////////
// 2021/5/27 generation functions

//       y+(0)
// x-(1) z+(2) x+(3) z-(4)
//       y-(5)
void GenerateCube(Mesh & mesh, float scale, bool generate_tex_coord) {

	mesh.clear();

	mesh.vertex_list.resize(8);
	mesh.vertex_list[0] << -scale, -scale, scale;
	mesh.vertex_list[1] << scale, -scale, scale;
	mesh.vertex_list[2] << scale, scale, scale;
	mesh.vertex_list[3] << -scale, scale, scale;
	
	mesh.vertex_list[4] << -scale, -scale, -scale;
	mesh.vertex_list[5] << scale, -scale, -scale;
	mesh.vertex_list[6] << scale, scale, -scale;
	mesh.vertex_list[7] << -scale, scale, -scale;
	//
	if (generate_tex_coord) {
		mesh.vertex_texture_list.resize(14);
		const float x2 = 0.66666667f;
		const float x1 = 0.33333333f;

		mesh.vertex_texture_list[0] << 0.25f, 1.f;
		mesh.vertex_texture_list[1] << 0.5f, 1.f;

		mesh.vertex_texture_list[2] << 0.f, x2;
		mesh.vertex_texture_list[3] << 0.25f, x2;
		mesh.vertex_texture_list[4] << 0.5f, x2;
		mesh.vertex_texture_list[5] << 0.75f, x2;
		mesh.vertex_texture_list[6] << 1.f, x2;

		mesh.vertex_texture_list[7] << 0.f, x1;
		mesh.vertex_texture_list[8] << 0.25f, x1;
		mesh.vertex_texture_list[9] << 0.5f, x1;
		mesh.vertex_texture_list[10] << 0.75f, x1;
		mesh.vertex_texture_list[11] << 1.f, x1;

		mesh.vertex_texture_list[12] << 0.25f, 0.f;
		mesh.vertex_texture_list[13] << 0.5f, 0.f;
	}

	mesh.face_list.resize(6);
	mesh.face_normal_list.resize(6);
	for (int i = 0; i < 6; i++) {
		mesh.face_list[i].vertex_count = 4; 
		mesh.face_list[i].material_index = 0;

		mesh.face_list[i].normal_index[0] = -1;
		mesh.face_list[i].normal_index[1] = -1;
		mesh.face_list[i].normal_index[2] = -1;
		mesh.face_list[i].normal_index[3] = -1;
		
		mesh.face_list[i].texture_index[0] = -1;
		mesh.face_list[i].texture_index[1] = -1;
		mesh.face_list[i].texture_index[2] = -1;
		mesh.face_list[i].texture_index[3] = -1;
	}
	mesh.face_list[0].vertex_index[0] = 7;
	mesh.face_list[0].vertex_index[1] = 3;
	mesh.face_list[0].vertex_index[2] = 2;
	mesh.face_list[0].vertex_index[3] = 6;

	mesh.face_list[1].vertex_index[0] = 7;
	mesh.face_list[1].vertex_index[1] = 4;
	mesh.face_list[1].vertex_index[2] = 0;
	mesh.face_list[1].vertex_index[3] = 3;

	mesh.face_list[2].vertex_index[0] = 0;
	mesh.face_list[2].vertex_index[1] = 1;
	mesh.face_list[2].vertex_index[2] = 2;
	mesh.face_list[2].vertex_index[3] = 3;

	mesh.face_list[3].vertex_index[0] = 2;
	mesh.face_list[3].vertex_index[1] = 1;
	mesh.face_list[3].vertex_index[2] = 5;
	mesh.face_list[3].vertex_index[3] = 6;

	mesh.face_list[4].vertex_index[0] = 7;
	mesh.face_list[4].vertex_index[1] = 6;
	mesh.face_list[4].vertex_index[2] = 5;
	mesh.face_list[4].vertex_index[3] = 4;

	mesh.face_list[5].vertex_index[0] = 0;
	mesh.face_list[5].vertex_index[1] = 4;
	mesh.face_list[5].vertex_index[2] = 5;
	mesh.face_list[5].vertex_index[3] = 1;
	//
	mesh.face_normal_list[0] << 0.f, 1.f, 0.f;
	mesh.face_normal_list[1] << -1.f, 0.f, 0.f;
	mesh.face_normal_list[2] << 0.f, 0.f, 1.f;
	mesh.face_normal_list[3] << 1.f, 0.f, 0.f;
	mesh.face_normal_list[4] << 0.f, 0.f, -1.f;
	mesh.face_normal_list[5] << 0.f, -1.f, 0.f;
	if (generate_tex_coord) {
		mesh.face_list[0].texture_index[0] = 0;
		mesh.face_list[0].texture_index[1] = 3;
		mesh.face_list[0].texture_index[2] = 4;
		mesh.face_list[0].texture_index[3] = 1;

		mesh.face_list[1].texture_index[0] = 2;
		mesh.face_list[1].texture_index[1] = 7;
		mesh.face_list[1].texture_index[2] = 8;
		mesh.face_list[1].texture_index[3] = 3;

		mesh.face_list[2].texture_index[0] = 8;
		mesh.face_list[2].texture_index[1] = 9;
		mesh.face_list[2].texture_index[2] = 4;
		mesh.face_list[2].texture_index[3] = 3;

		mesh.face_list[3].texture_index[0] = 4;
		mesh.face_list[3].texture_index[1] = 9;
		mesh.face_list[3].texture_index[2] = 10;
		mesh.face_list[3].texture_index[3] = 5;

		mesh.face_list[4].texture_index[0] = 6;
		mesh.face_list[4].texture_index[1] = 5;
		mesh.face_list[4].texture_index[2] = 10;
		mesh.face_list[4].texture_index[3] = 11;

		mesh.face_list[5].texture_index[0] = 8;
		mesh.face_list[5].texture_index[1] = 12;
		mesh.face_list[5].texture_index[2] = 13;
		mesh.face_list[5].texture_index[3] = 9;	
	}
}


// default R = 1
// H = 2, +1,-1
// init with +z oriented
void GenerateCylinder(Mesh & mesh, int N, float scaleR, float scaleH, bool generate_tex_coord)
{
	mesh.clear();

	if (generate_tex_coord) {
		printf("GenerateCylinder tex coord TODO!\n");
	}

	mesh.vertex_list.resize(2 * N + 2);
	mesh.vertex_normal_list.resize(N + 2);
	for (int i = 0; i < N; i++) {
		float t = float(i) / float(N) * 3.14159265358979323846f * 2;
		mesh.vertex_list[i] << scaleR * cos(t), scaleR * sin(t), scaleH;
		mesh.vertex_list[N + i] << scaleR * cos(t), scaleR * sin(t), -scaleH;
		mesh.vertex_normal_list[i]<< cos(t), sin(t), 0.f;
	}
	mesh.vertex_list[2 * N + 0] << 0, 0, scaleH;
	mesh.vertex_list[2 * N + 1] << 0, 0, -scaleH;
	mesh.vertex_normal_list[N + 0] << 0, 0, 1.f;
	mesh.vertex_normal_list[N + 1] << 0, 0, -1.f;

	mesh.face_list.resize(3 * N);
	for (int i = 0; i < 3 * N; i++) {
		mesh.face_list[i].vertex_count = (i < 2 * N) ? 3 : 4;
		mesh.face_list[i].material_index = 0;
		if (i < N) {
			mesh.face_list[i].vertex_index[0] = i;
			mesh.face_list[i].vertex_index[1] = (i + 1) % N;
			mesh.face_list[i].vertex_index[2] = 2 * N + 0;

			mesh.face_list[i].normal_index[0] = N + 0;
			mesh.face_list[i].normal_index[1] = N + 0;
			mesh.face_list[i].normal_index[2] = N + 0;
		}
		else if (i < 2 * N) {
			mesh.face_list[i].vertex_index[0] = N + (i + 1) % N;
			mesh.face_list[i].vertex_index[1] = N + i % N;
			mesh.face_list[i].vertex_index[2] = 2 * N + 1;

			mesh.face_list[i].normal_index[0] = N + 1;
			mesh.face_list[i].normal_index[1] = N + 1;
			mesh.face_list[i].normal_index[2] = N + 1;
		}
		else {
			mesh.face_list[i].vertex_index[0] = N + i % N;
			mesh.face_list[i].vertex_index[1] = N + (i + 1) % N;
			mesh.face_list[i].vertex_index[2] = (i + 1) % N;
			mesh.face_list[i].vertex_index[3] = i % N;

			mesh.face_list[i].normal_index[0] = i % N;
			mesh.face_list[i].normal_index[1] = i % N;
			mesh.face_list[i].normal_index[2] = i % N;
			mesh.face_list[i].normal_index[3] = i % N;
		}
		mesh.face_list[i].texture_index[0] = -1;
		mesh.face_list[i].texture_index[1] = -1;
		mesh.face_list[i].texture_index[2] = -1;
		mesh.face_list[i].texture_index[3] = -1;
	}
}
