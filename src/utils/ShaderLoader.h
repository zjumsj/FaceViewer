/* x64 edition */

//#define GLEW_STATIC
#include <GL/glew.h>

#ifdef _USE_GLUT
//#define FREEGLUT_STATIC
#include <GL/glut.h>
#endif

#include <string>
// msj 2021/3 add geometry shader support
#include <vector>

namespace ShaderLoader{
	
	//GLuint gltLoadShaderPairWithAttributes(int VertexProgLength,const char * VertexProg,int FragmentProgLength,const char * FragmentProg,...);
	//GLuint gltLoadShaderPairWithAttributes(const char *szVertexProg, const char *szFragmentProg, ...);
	GLuint gltLoadShaderPairWithAttributes(
		int VertexProgLength,
		const char * VertexProg,
		int FragmentProgLength,
		const char * FragmentProg,
		const std::vector<int> * attribindex = NULL,
		const std::vector<std::string> * attribname = NULL
	);

	GLuint gltLoadShaderPairWithAttributes(
		const char * szVertexProg,
		const char * szFragmentProg,
		const std::vector<int> * attribindex = NULL,
		const std::vector<std::string> * attribname = NULL
	);
	


	GLuint gltLoadShaderPairWithAttributesAdv(
		int VertexProgLength,
		const char * VertexProg,
		int FragmentProgLength,
		const char * FragmentProg,
		const std::vector<int> * attribindex = NULL,
		const std::vector<std::string> * attribname = NULL,
		int GeometryProgLength = 0,
		const char * GeometryProg = NULL,
		const std::vector<char *> * transformFeedbackVarying = NULL,
		int interleaved = 0
	);

	GLuint gltLoadShaderPairWithAttributesAdv(
		const char * szVertexProg,
		const char * szFragmentProg,
		const std::vector<int> * attribindex = NULL,
		const std::vector<std::string> * attribname = NULL,
		const char * szGeometryProg = NULL,
		const std::vector<char *> * transformFeedbackVarying = NULL,
		int interleaved = 0
	);

	//write a text->binary converter
	bool gltText2CharArray(const char * loadfile,const char * savefile);
	bool gltText2CharArray(const char * VertexProgArrayName,const char * FragmentProgArrayName,const char * szVertexProg,const char * szFragmentProg,const char * savefile);
	
	//DEBUG
	bool PrintCompileInfo(GLuint hShader);
};