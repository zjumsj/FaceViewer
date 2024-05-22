#include "ShaderLoader.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

namespace ShaderLoader{
	//const int MAX_SHADER_LENGTH=8192;
	const int MAX_SHADER_LENGTH = 16384;
	GLubyte shaderText[MAX_SHADER_LENGTH];

	//////////////////////////////////////////////////////////////////////////
	// Load the shader from the source text
	void gltLoadShaderSrc(const char *szShaderSrc, GLuint shader)
	{
		GLchar *fsStringPtr[1];

		fsStringPtr[0] = (GLchar *)szShaderSrc;
		glShaderSource(shader, 1, (const GLchar **)fsStringPtr, NULL);
	}


	////////////////////////////////////////////////////////////////
	// Load the shader from the specified file. Returns false if the
	// shader could not be loaded
	bool gltLoadShaderFile(const char *szFile, GLuint shader)
	{
		GLint shaderLength = 0;
		FILE *fp;
	
		// Open the shader file
		//fopen_s(&fp,szFile, "r");
		fp = fopen(szFile, "r");
		if(fp != NULL)
		{
			// See how long the file is
			while (fgetc(fp) != EOF)
				shaderLength++;
		
			// Allocate a block of memory to send in the shader
			assert(shaderLength < MAX_SHADER_LENGTH);   // make me bigger!
			if(shaderLength > MAX_SHADER_LENGTH)
			{
				fclose(fp);
				return false;
			}
		
			// Go back to beginning of file
			rewind(fp);
		
			// Read the whole file in
			if (shaderText != NULL)
				fread(shaderText, 1, shaderLength, fp);
		
			// Make sure it is null terminated and close the file
			shaderText[shaderLength] = '\0';
			fclose(fp);
		}
		else
			return false;    
	
		// Load the string
		gltLoadShaderSrc((const char *)shaderText, shader);
    	return true;
	}   
	/////////////////////////////////////////////////////////////////
	// Load a pair of shaders, compile, and link together. Specify the complete
	// source char pointer for each shader. After the shader names, specify the indexes
	// of attributes, and relevant parameter names in shader.
	GLuint gltLoadShaderPairWithAttributes(
		int VertexProgLength,
		const char * VertexProg,
		int FragmentProgLength,
		const char * FragmentProg,
		const std::vector<int> * attribindex ,
		const std::vector<std::string> * attribname 
	)
	{
		// Temporary Shader objects
		GLuint hVertexShader;
		GLuint hFragmentShader;
		GLuint hReturn = 0;
		GLint testVal;

		// Create shader objects
		hVertexShader = glCreateShader(GL_VERTEX_SHADER);// build the shader by type
		hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		
		GLchar * ProgPointer[1];
		ProgPointer[0] = (GLchar *)VertexProg;//Set Vertex Src
		if(VertexProgLength<=0)
			glShaderSource(hVertexShader,1,(const GLchar **)ProgPointer,NULL);
		else
			glShaderSource(hVertexShader,1,(const GLchar **)ProgPointer,&VertexProgLength);
		ProgPointer[0] = (GLchar *)FragmentProg;//Set Fragment Src
		if(FragmentProgLength<=0)
			glShaderSource(hFragmentShader,1,(const GLchar **)ProgPointer,NULL);
		else
			glShaderSource(hFragmentShader,1,(const GLchar **)ProgPointer,&FragmentProgLength);

		// Compile them both
		glCompileShader(hVertexShader);
		glCompileShader(hFragmentShader);
    
		// Check for errors in vertex shader
		glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
		if(testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hVertexShader, 1024, NULL, infoLog);
			fprintf(stderr, "The vertex shader failed to compile with the following error:\n%s\n%s\n", VertexProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			return (GLuint)NULL;
		}
   
		// Check for errors in fragment shader
		glGetShaderiv(hFragmentShader, GL_COMPILE_STATUS, &testVal);
		if(testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hFragmentShader, 1024, NULL, infoLog);
			fprintf(stderr, "The fragment shader failed to compile with the following error:\n%s\n%s\n", FragmentProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			return (GLuint)NULL;
		}
	    
		// Create the final program object, and attach the shaders
		hReturn = glCreateProgram();
		glAttachShader(hReturn, hVertexShader);
		glAttachShader(hReturn, hFragmentShader);
	
		// Now, we need to bind the attribute names to their specific locations
		// List of attributes
		// Iterate over this argument list
		if (attribindex && attribname) {
			int iArgCount = (*attribindex).size() > (*attribname).size() ? (*attribname).size() : (*attribindex).size();
			const char * szNextArg;
			int index;
			for (int i = 0; i < iArgCount; i++) {
				szNextArg = (*attribname)[i].c_str();
				index = (*attribindex)[i];
				glBindAttribLocation(hReturn, index, szNextArg);
			}
		}

		// Attempt to link    
		glLinkProgram(hReturn);
	
		// These are no longer needed
		glDeleteShader(hVertexShader);
		glDeleteShader(hFragmentShader);  
    
		// Make sure link worked too
		glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
		if(testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetProgramInfoLog(hReturn, 1024, NULL, infoLog);
			fprintf(stderr,"The programs %s and %s failed to link with the following errors:\n%s\n",
				VertexProg, FragmentProg, infoLog);
			glDeleteProgram(hReturn);
			return (GLuint)NULL;
		}
    
		// All done, return our ready to use shader program
		return hReturn;  
	}


	/////////////////////////////////////////////////////////////////
	// Load a pair of shaders, compile, and link together. Specify the complete
	// source text for each shader. After the shader names, specify the indexes
	// of attributes, and relevant parameter names in shader.
	GLuint gltLoadShaderPairWithAttributes(
		const char * szVertexProg,
		const char * szFragmentProg,
		const std::vector<int> * attribindex,
		const std::vector<std::string> * attribname
	)
	{
		// Temporary Shader objects
		GLuint hVertexShader;
		GLuint hFragmentShader;
		GLuint hReturn = 0;
		GLint testVal;

		// Create shader objects
		hVertexShader = glCreateShader(GL_VERTEX_SHADER);// build the shader by type
		hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

		// Load them. If fail clean up and return null
		// Vertex Program
		if (gltLoadShaderFile(szVertexProg, hVertexShader) == false)
		{
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			fprintf(stderr, "The shader at %s could not be found.\n", szVertexProg);
			return (GLuint)NULL;
		}

		// Fragment Program
		if (gltLoadShaderFile(szFragmentProg, hFragmentShader) == false)
		{
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			fprintf(stderr, "The shader at %s  could not be found.\n", szFragmentProg);
			return (GLuint)NULL;
		}

		// Compile them both
		glCompileShader(hVertexShader);
		glCompileShader(hFragmentShader);

		// Check for errors in vertex shader
		glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hVertexShader, 1024, NULL, infoLog);
			fprintf(stderr, "The shader at %s failed to compile with the following error:\n%s\n", szVertexProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			return (GLuint)NULL;
		}

		// Check for errors in fragment shader
		glGetShaderiv(hFragmentShader, GL_COMPILE_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hFragmentShader, 1024, NULL, infoLog);
			fprintf(stderr, "The shader at %s failed to compile with the following error:\n%s\n", szFragmentProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			return (GLuint)NULL;
		}


		// Create the final program object, and attach the shaders
		hReturn = glCreateProgram();
		glAttachShader(hReturn, hVertexShader);
		glAttachShader(hReturn, hFragmentShader);


		// Now, we need to bind the attribute names to their specific locations
		// List of attributes
		// Iterate over this argument list
		if (attribindex && attribname) {
			int iArgCount = (*attribindex).size() > (*attribname).size() ? (*attribname).size() : (*attribindex).size();
			const char * szNextArg;
			int index;
			for (int i = 0; i < iArgCount; i++) {
				szNextArg = (*attribname)[i].c_str();
				index = (*attribindex)[i];
				glBindAttribLocation(hReturn, index, szNextArg);
			}
		}

		// Attempt to link    
		glLinkProgram(hReturn);

		// These are no longer needed
		glDeleteShader(hVertexShader);
		glDeleteShader(hFragmentShader);

		// Make sure link worked too
		glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetProgramInfoLog(hReturn, 1024, NULL, infoLog);
			fprintf(stderr, "The programs %s and %s failed to link with the following errors:\n%s\n",
				szVertexProg, szFragmentProg, infoLog);
			glDeleteProgram(hReturn);
			return (GLuint)NULL;
		}

		// All done, return our ready to use shader program
		return hReturn;
	}


	/////////////////////////////////////////////////////////////////
	// An advanced edition of gltLoadShaderPairWithAttributes
	// support geometry shader + interleaved transform feedback
	GLuint gltLoadShaderPairWithAttributesAdv(
		int VertexProgLength,
		const char * VertexProg,
		int FragmentProgLength,
		const char * FragmentProg,
		const std::vector<int> * attribindex,
		const std::vector<std::string> * attribname,
		int GeometryProgLength,
		const char * GeometryProg,
		const std::vector<char *> * transformFeedbackVarying,
		int interleaved
	)
	{
		// Temporary Shader objects
		GLuint hVertexShader;
		GLuint hFragmentShader;
		GLuint hGeometryShader;
		GLuint hReturn = 0;
		GLint testVal;

		// Create shader objects
		hVertexShader = glCreateShader(GL_VERTEX_SHADER);// build the shader by type
		hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (GeometryProg) hGeometryShader = glCreateShader(GL_GEOMETRY_SHADER);

		GLchar * ProgPointer[1];
		ProgPointer[0] = (GLchar *)VertexProg;//Set Vertex Src
		if (VertexProgLength <= 0)
			glShaderSource(hVertexShader, 1, (const GLchar **)ProgPointer, NULL);
		else
			glShaderSource(hVertexShader, 1, (const GLchar **)ProgPointer, &VertexProgLength);
		
		ProgPointer[0] = (GLchar *)FragmentProg;//Set Fragment Src
		if (FragmentProgLength <= 0)
			glShaderSource(hFragmentShader, 1, (const GLchar **)ProgPointer, NULL);
		else
			glShaderSource(hFragmentShader, 1, (const GLchar **)ProgPointer, &FragmentProgLength);
		
		if (GeometryProg) {
			ProgPointer[0] = (GLchar *)GeometryProg; //Set Geometry Src
			if (GeometryProgLength <= 0)
				glShaderSource(hGeometryShader, 1, (const GLchar**)ProgPointer, NULL);
			else
				glShaderSource(hGeometryShader, 1, (const GLchar **)ProgPointer, &GeometryProgLength);
		}

		// Compile them both
		glCompileShader(hVertexShader);
		glCompileShader(hFragmentShader);
		if (GeometryProg) glCompileShader(hGeometryShader);

		// Check for errors in vertex shader
		glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hVertexShader, 1024, NULL, infoLog);
			fprintf(stderr, "The vertex shader failed to compile with the following error:\n%s\n%s\n", VertexProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			if (GeometryProg) glDeleteShader(hGeometryShader);
			return (GLuint)NULL;
		}

		// Check for errors in fragment shader
		glGetShaderiv(hFragmentShader, GL_COMPILE_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hFragmentShader, 1024, NULL, infoLog);
			fprintf(stderr, "The fragment shader failed to compile with the following error:\n%s\n%s\n", FragmentProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			if (GeometryProg) glDeleteShader(hGeometryShader);
			return (GLuint)NULL;
		}

		// Check for errors in geometry shader
		if (GeometryProg) {
			glGetShaderiv(hGeometryShader, GL_COMPILE_STATUS, &testVal);
			if (testVal == GL_FALSE)
			{
				char infoLog[1024];
				glGetShaderInfoLog(hGeometryShader, 1024, NULL, infoLog);
				fprintf(stderr, "The geometry shader failed to compile with the following error:\n%s\n%s\n", GeometryProg, infoLog);
				glDeleteShader(hVertexShader);
				glDeleteShader(hFragmentShader);
				glDeleteShader(hGeometryShader);
				return (GLuint)NULL;
			}
		}

		// Create the final program object, and attach the shaders
		hReturn = glCreateProgram();
		glAttachShader(hReturn, hVertexShader);
		glAttachShader(hReturn, hFragmentShader);
		if (GeometryProg) glAttachShader(hReturn, hGeometryShader);

		// Now, we need to bind the attribute names to their specific locations
		// List of attributes
		// Iterate over this argument list
		if (attribindex && attribname) {
			int iArgCount = (*attribindex).size() > (*attribname).size() ? (*attribname).size() : (*attribindex).size();
			const char * szNextArg;
			int index;
			for (int i = 0; i < iArgCount; i++) {
				szNextArg = (*attribname)[i].c_str();
				index = (*attribindex)[i];
				glBindAttribLocation(hReturn, index, szNextArg);
			}
		}
		// add transform feedback varyings 
		if (transformFeedbackVarying) {
			GLenum bufferMode = interleaved ? GL_INTERLEAVED_ATTRIBS : GL_SEPARATE_ATTRIBS;
			glTransformFeedbackVaryings(hReturn, (int)transformFeedbackVarying->size(), (*transformFeedbackVarying).data(), bufferMode);
		}

		// Attempt to link    
		glLinkProgram(hReturn);

		// These are no longer needed
		glDeleteShader(hVertexShader);
		glDeleteShader(hFragmentShader);
		if (GeometryProg) glDeleteShader(hGeometryShader);

		// Make sure link worked too
		glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetProgramInfoLog(hReturn, 1024, NULL, infoLog);
			if (GeometryProg) {
				fprintf(stderr, "The programs %s, %s and %s failed to link with the following errors:\n%s\n",
					VertexProg, GeometryProg, FragmentProg, infoLog);
			}
			else {
				fprintf(stderr, "The programs %s and %s failed to link with the following errors:\n%s\n",
					VertexProg, FragmentProg, infoLog);
			}
			glDeleteProgram(hReturn);
			return (GLuint)NULL;
		}

		// All done, return our ready to use shader program
		return hReturn;
	}

	/////////////////////////////////////////////////////////////////
	// An advanced edition of gltLoadShaderPairWithAttributes
	// support geometry shader + interleaved transform feedback
	GLuint gltLoadShaderPairWithAttributesAdv(
		const char * szVertexProg,
		const char * szFragmentProg,
		const std::vector<int> * attribindex,
		const std::vector<std::string> * attribname,
		const char * szGeometryProg,
		const std::vector<char *> * transformFeedbackVarying,
		int interleaved
	)
	{
		// Temporary Shader objects
		GLuint hVertexShader;
		GLuint hFragmentShader;
		GLuint hGeometryShader;
		GLuint hReturn = 0;
		GLint testVal;

		// Create shader objects
		hVertexShader = glCreateShader(GL_VERTEX_SHADER);// build the shader by type
		hFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (szGeometryProg) hGeometryShader = glCreateShader(GL_GEOMETRY_SHADER);

		// Load them. If fail clean up and return null
		// Vertex Program
		if (gltLoadShaderFile(szVertexProg, hVertexShader) == false)
		{
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			if (szGeometryProg) glDeleteShader(hGeometryShader);
			fprintf(stderr, "The shader at %s could not be found.\n", szVertexProg);
			return (GLuint)NULL;
		}

		// Fragment Program
		if (gltLoadShaderFile(szFragmentProg, hFragmentShader) == false)
		{
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			if (szGeometryProg) glDeleteShader(hGeometryShader);
			fprintf(stderr, "The shader at %s  could not be found.\n", szFragmentProg);
			return (GLuint)NULL;
		}

		// Geometry Program
		if (szGeometryProg) {
			if (gltLoadShaderFile(szGeometryProg, hGeometryShader) == false)
			{
				glDeleteShader(hVertexShader);
				glDeleteShader(hFragmentShader);
				glDeleteShader(hGeometryShader);
				fprintf(stderr, "The shader at %s  could not be found.\n", szGeometryProg);
				return (GLuint)NULL;
			}
		}

		// Compile them both
		glCompileShader(hVertexShader);
		glCompileShader(hFragmentShader);
		if (szGeometryProg) glCompileShader(hGeometryShader);

		// Check for errors in vertex shader
		glGetShaderiv(hVertexShader, GL_COMPILE_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hVertexShader, 1024, NULL, infoLog);
			fprintf(stderr, "The shader at %s failed to compile with the following error:\n%s\n", szVertexProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			if (szGeometryProg) glDeleteShader(hGeometryShader);
			return (GLuint)NULL;
		}

		// Check for errors in fragment shader
		glGetShaderiv(hFragmentShader, GL_COMPILE_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(hFragmentShader, 1024, NULL, infoLog);
			fprintf(stderr, "The shader at %s failed to compile with the following error:\n%s\n", szFragmentProg, infoLog);
			glDeleteShader(hVertexShader);
			glDeleteShader(hFragmentShader);
			if (szGeometryProg)glDeleteProgram(hGeometryShader);
			return (GLuint)NULL;
		}

		// Check for errors in geometry shader
		if (szGeometryProg) {
			glGetShaderiv(hGeometryShader, GL_COMPILE_STATUS, &testVal);
			if (testVal == GL_FALSE)
			{
				char infoLog[1024];
				glGetShaderInfoLog(hGeometryShader, 1024, NULL, infoLog);
				fprintf(stderr, "The shader at %s failed to compile with the following error:\n%s\n", szGeometryProg, infoLog);
				glDeleteShader(hVertexShader);
				glDeleteShader(hFragmentShader);
				glDeleteShader(hGeometryShader);
				return (GLuint)NULL;
			}
		}

		// Create the final program object, and attach the shaders
		hReturn = glCreateProgram();
		glAttachShader(hReturn, hVertexShader);
		glAttachShader(hReturn, hFragmentShader);
		if (szGeometryProg) glAttachShader(hReturn, hGeometryShader);


		// Now, we need to bind the attribute names to their specific locations
		// List of attributes
		// Iterate over this argument list
		if (attribindex && attribname) {
			int iArgCount = (*attribindex).size() > (*attribname).size() ? (*attribname).size() : (*attribindex).size();
			const char * szNextArg;
			int index;
			for (int i = 0; i < iArgCount; i++) {
				szNextArg = (*attribname)[i].c_str();
				index = (*attribindex)[i];
				glBindAttribLocation(hReturn, index, szNextArg);
			}
		}
		// add transform feedback varyings 
		if (transformFeedbackVarying) {
			GLenum bufferMode = interleaved ? GL_INTERLEAVED_ATTRIBS : GL_SEPARATE_ATTRIBS;
			glTransformFeedbackVaryings(hReturn, (int)transformFeedbackVarying->size(), (*transformFeedbackVarying).data(), bufferMode);
		}

		// Attempt to link    
		glLinkProgram(hReturn);

		// These are no longer needed
		glDeleteShader(hVertexShader);
		glDeleteShader(hFragmentShader);
		if (szGeometryProg) glDeleteShader(hGeometryShader);

		// Make sure link worked too
		glGetProgramiv(hReturn, GL_LINK_STATUS, &testVal);
		if (testVal == GL_FALSE)
		{
			char infoLog[1024];
			glGetProgramInfoLog(hReturn, 1024, NULL, infoLog);
			if (szGeometryProg) {
				fprintf(stderr, "The programs %s, %s and %s failed to link with the following errors:\n%s\n",
					szVertexProg, szGeometryProg, szFragmentProg, infoLog);
			}
			else {
				fprintf(stderr, "The programs %s and %s failed to link with the following errors:\n%s\n",
					szVertexProg, szFragmentProg, infoLog);
			}
			glDeleteProgram(hReturn);
			return (GLuint)NULL;
		}

		// All done, return our ready to use shader program
		return hReturn;
	}


	/////////////////////////////////////////////////////////////////
	// Load a shader in text format and transfer it to the format 
	// which can be easily copied as the content of const char arraies.
	// The output is saved as txt and named 'savefile'
	bool gltText2CharArray(const char * loadfile,const char * savefile){
		//FILE * rfp; fopen_s(&rfp,loadfile,"r");
		//FILE * sfp; fopen_s(&sfp,savefile,"w");
		FILE * rfp = fopen(loadfile,"r");
		FILE * sfp = fopen(savefile,"w");
		
		if(rfp==NULL || sfp==NULL) {
			if(rfp) fclose(rfp);
			if(sfp) fclose(sfp);
			return false;
		}
		unsigned char inbuff[32];unsigned char * outfp;
		unsigned char outbuff[256];
		unsigned char temp[8];
		while(1){
			int n=fread(inbuff,sizeof(char),32,rfp);
			if(n==0) break;
			outfp = outbuff;
			for (int i=0;i<n;i++){
				if(inbuff[i]==32){//whitespace
					outfp[0]=inbuff[i]; outfp++;				
				}// It seems \r is auto-removed
				else if(inbuff[i]=='\n'){//\n
					outfp[0]='\\';outfp[1]='n';	outfp[2]='\\';
					//outfp[3]=0x0D; //It seems \r is auto-added
					outfp[3]=0x0A;//windows new line
					outfp+=4;
				}else if(inbuff[i]<=32){
					//sprintf_s((char *)temp,8,"%03o",inbuff[i]);
					sprintf((char *)temp,"%03o",inbuff[i]);
					outfp[0]='\\';
					int buflen = strlen((char *)temp);
					for(int j=0;j<buflen;j++)
						outfp[1+j]=temp[j];
					outfp += (buflen+1);
				}else if(inbuff[i]=='\\'){//should specially transform
					outfp[0]='\\'; outfp[1]='\\'; outfp+=2;
				}
				else{
					*outfp = inbuff[i]; outfp++;
				}
			}
			*outfp=0;
			fputs((char *)outbuff,sfp);
		}
		fclose(rfp);
		fclose(sfp);
		return true;
	}

	/////////////////////////////////////////////////////////////////
	// Load a pair of shaders in text format and transfer them to 
	// the format which can be easily copied as the content of
	// const char arraies. The output is saved under name 'savefile'
	// and is well-organized and can be directly included.
	bool gltText2CharArray(const char * VertexProgArrayName,const char * FragmentProgArrayName,const char * szVertexProg,const char * szFragmentProg,const char * savefile)
	{
		FILE * rfp[2];
		//fopen_s(&rfp[0],szVertexProg,"r");
		//fopen_s(&rfp[1],szFragmentProg,"r");
		//FILE * sfp; fopen_s(&sfp,savefile,"w");
		rfp[0] = fopen(szVertexProg, "r");
		rfp[1] = fopen(szFragmentProg, "r");
		FILE * sfp = fopen(savefile, "w");
		
		if(rfp[0]==NULL || rfp[1]==NULL || sfp==NULL) {
			if(rfp[0]) 
				fclose(rfp[0]);
			if(rfp[1]) 
				fclose(rfp[1]);
			if(sfp) 
				fclose(sfp);
			return false;
		}
		unsigned char inbuff[32];unsigned char * outfp;
		unsigned char outbuff[256];
		unsigned char temp[8];
		//fprintf_s(sfp,"// generated from:\n// vertex shader:%s\n// fragment shader:%s\n\n",szVertexProg,szFragmentProg);
		fprintf(sfp,"// generated from:\n// vertex shader:%s\n// fragment shader:%s\n\n",szVertexProg,szFragmentProg);
		for (int item=0;item<2;item++){
			if(item==0)
				//fprintf_s(sfp,"static char %s[]=\"\\\n",VertexProgArrayName);
				fprintf(sfp,"static char %s[]=\"\\\n",VertexProgArrayName);
			else if(item==1)
				//fprintf_s(sfp,"static char %s[]=\"\\\n",FragmentProgArrayName);
				fprintf(sfp,"static char %s[]=\"\\\n",FragmentProgArrayName);
			while(1){
				int n=fread(inbuff,sizeof(char),32,rfp[item]);
				if(n==0) break;
				outfp = outbuff;
				for (int i=0;i<n;i++){
					if(inbuff[i]==32){//whitespace
						outfp[0]=inbuff[i]; outfp++;				
					}// It seems \r is auto-removed
					else if(inbuff[i]=='\n'){//\n
						outfp[0]='\\';outfp[1]='n';	outfp[2]='\\';
						//outfp[3]=0x0D; //It seems \r is auto-added
						outfp[3]=0x0A;//windows new line
						outfp+=4;
					}else if(inbuff[i]<=32){
						//sprintf_s((char *)temp,8,"%03o",inbuff[i]);
						sprintf((char *)temp, "%03o", inbuff[i]);
						outfp[0]='\\';
						int buflen = strlen((char *)temp);
						for(int j=0;j<buflen;j++)
							outfp[1+j]=temp[j];
						outfp += (buflen+1);
					}else if(inbuff[i]=='\\'){//should specially transform
						outfp[0]='\\'; outfp[1]='\\'; outfp+=2;
					}
					else{
						*outfp = inbuff[i]; outfp++;
					}
				}
				*outfp=0;
				fputs((char *)outbuff,sfp);
			}
			//fprintf_s(sfp,"\";\n\n");//Print some Enter
			fprintf(sfp,"\";\n\n");//Print some Enter
		}
		fclose(rfp[0]);fclose(rfp[1]);
		fclose(sfp);
		return true;
	}



	//DEBUG 
	bool PrintCompileInfo(GLuint hShader)
	{
		int length;
		char * infopointer;
		glGetShaderiv(hShader,GL_INFO_LOG_LENGTH,&length);
		if(length){
			infopointer = new char[length];
			glGetShaderInfoLog(hShader,length,NULL,infopointer);
			printf("Info:\n%s\n",infopointer);
			delete [] infopointer;
			return true;
		}
		printf("Invalid hShader\n");
		return false;
	}
};