#include "Scene.h"
#include <time.h>
#include <sstream>
#include "OpenGL.h"
using namespace std;

clock_t startClock;
clock_t endClock;
void calculateFPS(clock_t sc, clock_t ec, GLFWwindow* wind)
{
    static float durationTime = 0.0f;
    static int fps = 0;
    durationTime += (float(ec - sc) / float(CLOCKS_PER_SEC));
    fps++;
    if (durationTime >= 0.3f)
    {
        ostringstream oss;
        oss << "Shadow Map : " << fps << " fps";
        glfwSetWindowTitle(wind, oss.str().c_str());
        durationTime = 0.0f;
        fps = 0;
    }
}
int main()
{
    const int winWidth = 720;
    const int winHeight = 720;

    OpenGL ogl(winWidth, winHeight, "cuda pipeline");
    glViewport(0, 0, winWidth, winHeight);
    glEnable(GL_DEPTH_TEST);
    Shader shader("./Shader/SceneVert.glsl", "./Shader/SceneFrag.glsl");
    float3 pos[4] = {
        -1,-1,0,
        1,-1,0,
        -1,1,0,
        1,1,0
    };
    uint index[6] = { 0,1,2,1,3,2 };
    uint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(pos), pos, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index), index, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
    glBindVertexArray(0);

    uint texture;
    ogl.createEmptyTexture(texture, winWidth, winHeight);

    clock_t startClock;
    clock_t endClock;
    std::cout << "start to initialize scene" << std::endl;
    startClock = clock();
	Scene scene(winWidth, winHeight);
    Material* red = new Material(DIFFUSE, make_float3(0.0f));
    red->kd = make_float3(0.63f, 0.065f, 0.05f);
    Material* green = new Material(DIFFUSE, make_float3(0.0f));
    green->kd = make_float3(0.14f, 0.45f, 0.091f);
    Material* white = new Material(DIFFUSE, make_float3(0.0f));
    white->kd = make_float3(0.725f, 0.71f, 0.68f);
    Material* light = new Material(DIFFUSE, (8.0f * make_float3(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 
        15.6f * make_float3(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * make_float3(0.737f + 0.642f, 0.737f + 0.159f, 0.737f)));
    light->kd = make_float3(0.65f);

    MeshTriangle floor("./models/cornellbox/floor.obj", white);
    MeshTriangle shortbox("./models/cornellbox/shortbox.obj", white);
    MeshTriangle tallbox("./models/cornellbox/tallbox.obj", white);
    MeshTriangle left("./models/cornellbox/left.obj", red);
    MeshTriangle right("./models/cornellbox/right.obj", green);
    MeshTriangle light_("./models/cornellbox/light.obj", light);

    scene.add(&floor);
    scene.add(&shortbox);
    scene.add(&tallbox);
    scene.add(&left);
    scene.add(&right);
    scene.add(&light_);
    scene.printInfo();
    endClock = clock();
    std::cout << "initial scene use time " << 
        (float)(endClock - startClock) / (float)CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << std::endl;
    std::cout << "start to build native BVH" << std::endl;
    startClock = clock();
    scene.buildBVH();
    endClock = clock();
    std::cout << "building native BHV use time " << 
        (float)(endClock - startClock) / (float)CLOCKS_PER_SEC << " seconds" << std::endl;

    std::cout << std::endl;
    std::cout << "start to upload data" << std::endl;
    startClock = clock();
    scene.uploadData();
    endClock = clock();
    std::cout << "upload data use time " <<
        (float)(endClock - startClock) / (float)CLOCKS_PER_SEC << " seconds" << std::endl;

    scene.registerOpenGL(texture);
    
    while (!ogl.windowShouldClose())
    {
        startClock = clock();
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shader.use();
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
       
        scene.draw();

        shader.setInt("tex", 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        //接收鼠标键盘等的操作
        glfwPollEvents();
        //使用双buffer防止画面撕裂，即防止前一帧绘制到一半的时候来新的数据，导致画面中两帧的内容被混在一起输出
        glfwSwapBuffers(ogl.getWindow());
        endClock = clock();
        calculateFPS(startClock, endClock, ogl.getWindow());
    }
    
    std::cout << std::endl;
    std::cout << "start to draw img" << std::endl;
    startClock = clock();
    scene.draw();
    endClock = clock();
    std::cout << "draw img use time " <<
        (float)(endClock - startClock) / (float)CLOCKS_PER_SEC << " seconds" << std::endl;
	return 0;
}