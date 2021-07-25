#version 330 core
out vec4 FragColor;

in vec2 texCoord;
uniform sampler2D tex;
void main()
{
	// ���µߵ�һ��
	FragColor = texture(tex, vec2(1.0f-texCoord.x,1.0f-texCoord.y));
	//FragColor = vec4(0.5f);
}