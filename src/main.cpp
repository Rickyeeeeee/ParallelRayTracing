// main.cpp
#include <core/core.h>
#include <opengl/opengl_utils.h>
#include <opengl/opengl_renderer.h>
#include <core/film.h>
#include <backend/cpu/renderer.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

// CUDA
#include <cuda_runtime.h>

// OptiX
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cstdio>
#include <iostream>

bool checkCUDA() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Failed to query device count: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    if (deviceCount == 0) {
        std::cerr << "[CUDA] No CUDA devices found.\n";
        return false;
    }

    std::cout << "[CUDA] Found " << deviceCount << " CUDA device(s).\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "  Using Device 0: " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";

    return true;
}

bool checkOptiX() {
    // OptiX uses a function table created by optixInit()
    OptixResult res = optixInit();

    if (res != OPTIX_SUCCESS) {
        std::cerr << "[OptiX] optixInit() failed. Error code = " << res << "\n";
        return false;
    }

    std::cout << "[OptiX] Successfully initialized OptiX.\n";
    return true;
}

int main() {

    if (!glfwInit()) {
        std::cerr << "[Error] Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    uint32_t windowWidth = 800;
    uint32_t windowHeight = 600;
    GLFWwindow* window =
        glfwCreateWindow(windowWidth, windowHeight, "System Check: OpenGL + CUDA + OptiX + ImGui", nullptr, nullptr);

    if (!window) {
        std::cerr << "[Error] Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[Error] Failed to initialize GLAD\n";
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL Version:   " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    // --------------------------
    // CUDA + OptiX checks
    // --------------------------
    bool cudaOK = checkCUDA();
    bool optixOK = checkOptiX();

    // --------------------------
    // Setup Dear ImGui
    // --------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    bool show_demo = false;



    // Initialization
    Film film{ windowWidth, windowHeight };
    float* rgbData = new float[windowWidth * windowHeight * 3];
    for (int y = 0; y < windowHeight; y++)
    {
        for (int x = 0; x < windowWidth; x++)
        {
            int idx = 3 * (y * windowWidth + x);
            rgbData[idx + 0] = 1.0f;
            rgbData[idx + 1] = 0.5f;
            rgbData[idx + 2] = 0.5f;
        }
    }
    
    film.AddSampleBuffer(rgbData);
    film.UpdateDisplay();

    OpenGLTextureRenderer openglRenderer{};
    Scene scene{};

    auto center = glm::vec3{ 5.0f, 5.0f, 8.0f };
    auto focus = glm::vec3{ 0.0f };
    Camera camera {
        center,
        glm::normalize(focus - center),
        static_cast<float>(windowWidth),
        static_cast<float>(windowHeight),
        100.0f
    };
    auto cpuRenderer = std::make_shared<CPURenderer>();
    cpuRenderer->Init(film, scene, camera);

    
    std::shared_ptr<OpenGLTexture> frame = std::make_shared<OpenGLTexture>(windowWidth, windowHeight);
    

    // --------------------------
    // Main Loop
    // --------------------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // System check window
        ImGui::Begin("System Diagnostics");

        ImGui::Text("OpenGL: %s", (const char*)glGetString(GL_VERSION));

        ImGui::Separator();
        ImGui::Text("CUDA Status:");
        ImGui::TextColored(cudaOK ? ImVec4(0,1,0,1) : ImVec4(1,0,0,1),
                           cudaOK ? "CUDA Loaded Successfully" : "CUDA Failed");

        ImGui::Separator();
        ImGui::Text("OptiX Status:");
        ImGui::TextColored(optixOK ? ImVec4(0,1,0,1) : ImVec4(1,0,0,1),
                           optixOK ? "OptiX Loaded Successfully" : "OptiX Failed");


        ImGui::End();


        // Render
        ImGui::Render();
        
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.15f, 0.18f, 0.22f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        cpuRenderer->ProgressiveRender();
        film.UpdateDisplay();
        frame->SetData(film.GetDisplayData());
        openglRenderer.Draw(*frame);
        

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwSwapInterval(1);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
