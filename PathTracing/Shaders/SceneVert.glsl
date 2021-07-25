#version 330 core
layout(location = 0) in vec3 aPos;
out vec2 texCoord;
void main()
{
	texCoord = aPos.xy;
	texCoord += 1.0f;
	texCoord /= 2.0f;
	gl_Position = vec4(aPos, 1.0);
}