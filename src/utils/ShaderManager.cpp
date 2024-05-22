#include "ShaderLoader.h"
#include "ShaderManager.h"
#include <stdarg.h>

ShaderManager::ShaderManager()
{
	current = ShaderMap.end(); 	
}

ShaderManager::~ShaderManager()
{
	for (std::map<std::string,ShaderInfo *>::iterator it= ShaderMap.begin(); it!=ShaderMap.end(); it++){
		glDeleteProgram(it->second->hShader);
		delete it->second;
		it->second = NULL;
	}
}

void ShaderManager::Clear()
{
	for (std::map<std::string,ShaderInfo *>::iterator it= ShaderMap.begin(); it!=ShaderMap.end(); it++){
		glDeleteProgram(it->second->hShader);
		delete  it->second;
		it->second = NULL;
	}
	ShaderMap.clear();
	current = ShaderMap.end();
}

bool ShaderManager::LoadFromFileAdv(const char * ShaderName, const char * VertexProg, const char * FragmentProg, const char * GeometryProg,
	const std::vector<int> & attribindex, const std::vector<std::string> & attribname, const std::vector<char *> & transformFeedbackVarying,
	int interleaved)
{
	const std::vector<char *> * p_feedback = transformFeedbackVarying.size() ? &transformFeedbackVarying : nullptr;
	GLuint flag = ShaderLoader::gltLoadShaderPairWithAttributesAdv(VertexProg, FragmentProg, &attribindex, &attribname ,GeometryProg,p_feedback, interleaved);
	if (!flag)
		return false;
	std::map<std::string, ShaderInfo *>::iterator it = ShaderMap.find(ShaderName);
	if (it == ShaderMap.end()) {//add new
		std::pair<std::string, ShaderInfo *> nelem(std::string(ShaderName), new ShaderInfo);
		nelem.second->hShader = flag;
		ShaderMap.insert(nelem);
	}
	else {//clean the old
		ShaderInfo * shaderinfo = it->second;
		glDeleteProgram(shaderinfo->hShader);
		shaderinfo->hShader = flag;
		shaderinfo->id.clear();
		shaderinfo->UniformParam.clear();
	}
	current = ShaderMap.find(std::string(ShaderName));//ReOrient
	return true;

}

bool ShaderManager::LoadFromFile(const char * ShaderName, const char * VertexProg, const char * FragmentProg, ...){
	va_list attributeList;
	va_start(attributeList, FragmentProg);
	int iArgCount = va_arg(attributeList, int);
	std::vector<int> temp_index(iArgCount);
	std::vector<std::string> temp_name(iArgCount);
	for (int i = 0; i < iArgCount; i++) {
		temp_index[i] = va_arg(attributeList, int);
		temp_name[i] = std::string(va_arg(attributeList, char*));
	}
	va_end(attributeList);
	GLuint flag = ShaderLoader::gltLoadShaderPairWithAttributes(VertexProg,FragmentProg,&temp_index,&temp_name);
	if(!flag)
		return false;
	std::map<std::string,ShaderInfo *>::iterator it=ShaderMap.find(ShaderName);
	if(it==ShaderMap.end()){//add new
		std::pair<std::string,ShaderInfo *> nelem(std::string(ShaderName),new ShaderInfo);
		nelem.second->hShader = flag;
		ShaderMap.insert(nelem);
	}
	else{//clean the old
		ShaderInfo * shaderinfo = it->second;
		glDeleteProgram(shaderinfo->hShader);
		shaderinfo->hShader = flag;
		shaderinfo->id.clear();
		shaderinfo->UniformParam.clear();
	}
	current = ShaderMap.find(std::string(ShaderName));//ReOrient
	return true;
}

bool ShaderManager::LoadFromCharArrayAdv(const char * ShaderName, const char * VertexProg, const char * FragmentProg, const char * GeometryProg,
	const std::vector<int> & attribindex, const std::vector<std::string> & attribname, const std::vector<char *> & transformFeedbackVarying,
	int interleaved )
{
	const std::vector<char *> * p_feedback = transformFeedbackVarying.size() ? &transformFeedbackVarying : nullptr;
	GLuint flag = ShaderLoader::gltLoadShaderPairWithAttributesAdv(0,VertexProg, 0,FragmentProg, &attribindex, &attribname, 0,GeometryProg, p_feedback, interleaved);
	if (!flag)
		return false;
	std::map<std::string, ShaderInfo *>::iterator it = ShaderMap.find(ShaderName);
	if (it == ShaderMap.end()) {//add new
		std::pair<std::string, ShaderInfo *> nelem(std::string(ShaderName), new ShaderInfo);
		nelem.second->hShader = flag;
		ShaderMap.insert(nelem);
	}
	else {//clean the old
		ShaderInfo * shaderinfo = it->second;
		glDeleteProgram(shaderinfo->hShader);
		shaderinfo->hShader = flag;
		shaderinfo->UniformParam.clear();//clean
		shaderinfo->id.clear();
	}
	current = ShaderMap.find(std::string(ShaderName));//ReOrient
	return true;
}

bool ShaderManager::LoadFromCharArray(const char * ShaderName, const char * VertexProg, const char * FragmentProg, ...){
	va_list attributeList;
	va_start(attributeList, FragmentProg);
	int iArgCount = va_arg(attributeList, int);
	std::vector<int> temp_index(iArgCount);
	std::vector<std::string> temp_name(iArgCount);
	for (int i = 0; i < iArgCount; i++) {
		temp_index[i] = va_arg(attributeList, int);
		temp_name[i] = std::string(va_arg(attributeList, char*));
	}
	va_end(attributeList);
	GLuint flag = ShaderLoader::gltLoadShaderPairWithAttributes(0,VertexProg,0,FragmentProg,&temp_index,&temp_name);
	if(!flag)
		return false;
	std::map<std::string,ShaderInfo *>::iterator it=ShaderMap.find(ShaderName);
	if(it==ShaderMap.end()){//add new
		std::pair<std::string,ShaderInfo *> nelem(std::string(ShaderName),new ShaderInfo);
		nelem.second->hShader = flag;
		ShaderMap.insert(nelem);
	}
	else{//clean the old
		ShaderInfo * shaderinfo = it->second;
		glDeleteProgram(shaderinfo->hShader);
		shaderinfo->hShader = flag;
		shaderinfo->UniformParam.clear();//clean
		shaderinfo->id.clear();
	}
	current = ShaderMap.find(std::string(ShaderName));//ReOrient
	return true;
}

bool ShaderManager::SetCurrentShader(const char * ShaderName){
	current = ShaderMap.find(std::string(ShaderName));
	if(current == ShaderMap.end())
		return false;
	else
		return true;
}

bool ShaderManager::UseCurrentShader(){
	if(current == ShaderMap.end())
		return false;
	glUseProgram(current->second->hShader);
	return true;
}

bool ShaderManager::UseDefaultShader(){
	glUseProgram(0);
	return true;
}

bool ShaderManager::DeleteCurrentShader(){
	if(current == ShaderMap.end())
		return false;
	glDeleteProgram(current->second->hShader);
	delete current->second;
	current->second = NULL;
	current = ShaderMap.erase(current);
	return true;
}

bool ShaderManager::SetUniformParams(const char * ParamName, DATA_TYPE type,int nElement,void * data){
	if(current == ShaderMap.end())
		return false;
	ShaderInfo * shaderinfo = current->second;
	const int nsize =shaderinfo->UniformParam.size();
	std::map<std::string,int>::iterator uit = shaderinfo->UniformParam.find(std::string(ParamName));
	GLint hParam;
	if(uit==shaderinfo->UniformParam.end()){//add new
		hParam = glGetUniformLocation(shaderinfo->hShader,ParamName);
		if(hParam == -1) return false;//failed get handle
		shaderinfo->id.resize(nsize+1);
		shaderinfo->id[nsize] = hParam;
		std::pair<std::string,int> elem(std::string(ParamName),nsize);
		shaderinfo->UniformParam.insert(elem);
	}else{
		int idx = uit->second;
		hParam = shaderinfo->id[idx];	
	}
	switch(type){
	case VEC1F:
		glUniform1fv(hParam,nElement,(GLfloat *)data);break;
	case VEC2F:
		glUniform2fv(hParam,nElement,(GLfloat *)data);break;
	case VEC3F:
		glUniform3fv(hParam,nElement,(GLfloat *)data);break;
	case VEC4F:
		glUniform4fv(hParam,nElement,(GLfloat *)data);break;
	case VEC1I:
		glUniform1iv(hParam,nElement,(GLint *)data);break;
	case VEC2I:
		glUniform2iv(hParam,nElement,(GLint *)data);break;
	case VEC3I:
		glUniform3iv(hParam,nElement,(GLint *)data);break;
	case VEC4I:
		glUniform4iv(hParam,nElement,(GLint *)data);break;
	case MAT2F:
		glUniformMatrix2fv(hParam,nElement,GL_FALSE,(GLfloat *)data); break;
	case MAT3F:
		glUniformMatrix3fv(hParam,nElement,GL_FALSE,(GLfloat *)data); break;
	case MAT4F:
		glUniformMatrix4fv(hParam,nElement,GL_FALSE,(GLfloat *)data); break;
	default:
		return false;	
	}
	return true;
}


void CheckFrameBufferState() {
	GLenum tmpStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (tmpStatus != GL_FRAMEBUFFER_COMPLETE)
		printf("Framebuffer bind fail!\n");
}
void GetError() {
	GLenum x = glGetError();
	switch (x) {
	case GL_INVALID_OPERATION:
		printf("GL_INVALID_OPERATION\n");
		break;
	case GL_INVALID_ENUM:
		printf("GL_INVALID_ENUM\n");
		break;
	case GL_INVALID_VALUE:
		printf("GL_INVALID_VALUE\n");
		break;
		// any more errors go here
	case GL_NO_ERROR:
		break;
	default:
		printf("Unknow error\n");
		break;
	}
}




