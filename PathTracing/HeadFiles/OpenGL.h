#pragma once
#include "Shader.h"
#include "GLFW/glfw3.h"
#include <string>
using namespace std;
class OpenGL
{
private:
	GLFWwindow* window;
	int winWidth;
	int winHeight;
	string winName;
public:
	void resize(int width, int height);
	GLFWwindow* getWindow() { return window; }
	int getWinWidth() { return winWidth; }
	int getWinHeight() { return winHeight; }
	OpenGL(int width, int height, string name);
	bool windowShouldClose();
	bool createEmptyTexture(unsigned int& texture, int texWidth, int texHeight);
};

