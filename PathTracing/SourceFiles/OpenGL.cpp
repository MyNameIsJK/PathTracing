#include "OpenGL.h"

void OpenGL::resize(int width, int height)
{
	winWidth = width;
	winHeight = height;
	glViewport(0, 0, winWidth, winHeight);
}

OpenGL::OpenGL(int width, int height, string name):
	winWidth(width),winHeight(height),winName(name)
{
	// 初始化glfw
	glfwInit();
	//给出三个提示 hint
	//给出使用的glfw主版本号，如glfw1.3主版本号是1，副版本号是3
	glfwInitHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	//给出glfw副版本号
	glfwInitHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	//给出glfw配置profile，早期是固定流水线，现在是可编程流水线
	glfwInitHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//创建glfw窗口
	window = glfwCreateWindow(winWidth, winHeight, winName.c_str(), NULL, NULL);

	if (window == NULL)
	{
		printf("failed to open the glfw window\n");
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);

	//初始化glew
	glewExperimental = true;
	if (!glewInit() == GLEW_OK)
	{
		glfwTerminate();
		printf("failed to init glew\n");
	}
}

bool OpenGL::windowShouldClose()
{
	return glfwWindowShouldClose(window);
}

bool OpenGL::createEmptyTexture(unsigned int& texture, int texWidth, int texHeight)
{
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);	
	return true;
}
