/* 
    x64 Edition 
*/

#pragma once
//#define GLEW_STATIC
#include <GL/glew.h>

#ifdef _USE_GLUT
//#define FREEGLUT_STATIC
#include <GL/glut.h>
#endif

#include <map>
#include <string>

#include <vector>

/*
	For uniform variable arrays, each element of the array is considered to be of the type indicated in the name of the command 
	(e.g., glUniform3f or glUniform3fv can be used to load a uniform variable array of type vec3).
	The number of elements of the uniform variable array to be modified is specified by count
*/

class ShaderManager{
public:
	enum DATA_TYPE{VEC1F,VEC2F,VEC3F,VEC4F,
		VEC1I,VEC2I,VEC3I,VEC4I,
		MAT2F,MAT3F,MAT4F};
public:
	ShaderManager();
	~ShaderManager();

	bool LoadFromFile(const char * ShaderName,const char * VertexProg,const char * FragmentProg,...);
	bool LoadFromCharArray(const char * ShaderName,const char * VertexProg,const char * FragmentProg,...);
	
	bool LoadFromFileAdv(const char * ShaderName, const char * VertexProg, const char * FragmentProg,const char * GeometryProg,
		const std::vector<int> & attribindex = {}, const std::vector<std::string> & attribname = {}, const std::vector<char *> & transformFeedbackVarying = {},
		int interleaved = 0	);
	bool LoadFromCharArrayAdv(const char * ShaderName, const char * VertexProg, const char * FragmentProg, const char * GeometryProg,
		const std::vector<int> & attribindex = {}, const std::vector<std::string> & attribname = {}, const std::vector<char *> & transformFeedbackVarying = {},
		int interleaved = 0 );


	bool SetUniformParams(const char * ParamName,DATA_TYPE type,int nElement,void * data) ;

	bool SetCurrentShader(const char * ShaderName);
	
	bool UseCurrentShader();
	bool UseDefaultShader();

	bool DeleteCurrentShader();

	void Clear();

	inline int ShaderNum() const { return (int)ShaderMap.size(); }

private:

	struct ShaderInfo{
		std::map<std::string,int> UniformParam;
		std::vector<GLint> id;
		GLuint hShader;
	};
	std::map<std::string,ShaderInfo *> ShaderMap;
	std::map<std::string,ShaderInfo *>::iterator current;
};

void GetError();
void CheckFrameBufferState();
