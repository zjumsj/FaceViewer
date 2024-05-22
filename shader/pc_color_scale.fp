#version 330

// input
in vec4 gl_FragCoord;

// output
out vec4 vFragColor;


uniform float scale = 1.f;
smooth in vec3 c;

void main(void){
	vFragColor = vec4(c * scale,1.f);
}
