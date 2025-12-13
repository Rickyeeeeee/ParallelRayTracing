// main.cpp
#include <core/core.h>
#include <opengl/opengl_utils.h>
#include <opengl/opengl_renderer.h>
#include <core/film.h>
#include <backend/cpu/renderer.h>
#include <backend/cuda_megakernel/renderer.h>
#include <backend/cuda_wavefront/renderer.h>
#include <backend/optix/renderer.h>

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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

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

    uint32_t windowWidth = 1200;
    uint32_t windowHeight = 900;
    GLFWwindow* window =
        glfwCreateWindow(windowWidth, windowHeight, "System Check: OpenGL + CUDA + OptiX + ImGui", nullptr, nullptr);

    if (!window) {
        std::cerr << "[Error] Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

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

    // Initialization
    Film film{ windowWidth, windowHeight };
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

    struct RendererOption
    {
        const char* Label;
        std::shared_ptr<Renderer> Instance;
    };

    std::vector<RendererOption> rendererOptions;
    rendererOptions.push_back({ "CPU (std::async)", std::make_shared<CPURenderer>() });
    rendererOptions.push_back({ "CUDA Megakernel", std::make_shared<CudaMegakernelRenderer>() });
    rendererOptions.push_back({ "CUDA Wavefront", std::make_shared<CudaWavefrontRenderer>() });
    if (optixOK)
        rendererOptions.push_back({ "OptiX (dummy)", std::make_shared<OptixRenderer>() });

    for (auto& option : rendererOptions)
        option.Instance->Init(film, scene, camera);

    film.Clear();

    int selectedRenderer = 0;

    std::shared_ptr<OpenGLTexture> frame = std::make_shared<OpenGLTexture>(windowWidth, windowHeight);
    double lastRenderMs = 0.0;
    double lastFilmUpdateMs = 0.0;
    double lastUploadMs = 0.0;
    
    // Performance history for graphs
    constexpr size_t historySize = 100;
    std::vector<float> renderHistory(historySize, 0.0f);
    std::vector<float> filmUpdateHistory(historySize, 0.0f);
    std::vector<float> uploadHistory(historySize, 0.0f);
    std::vector<float> totalHistory(historySize, 0.0f);
    std::vector<float> fpsHistory(historySize, 0.0f);
    size_t historyIndex = 0;
    

    // --------------------------
    // Main Loop
    // --------------------------
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // System check window
        ImGui::Begin("System Diagnostics");

        ImGui::Text("FPS: %.3f", 1.0 / delta);

        ImGui::Text("OpenGL: %s", (const char*)glGetString(GL_VERSION));
        ImGui::Text("Timings (ms): render %.3f | film %.3f | upload/draw %.3f", lastRenderMs, lastFilmUpdateMs, lastUploadMs);

        ImGui::Separator();
        ImGui::Text("Performance Graphs");
        
        // Calculate statistics
        float maxTotal = 0.0f;
        for (float val : totalHistory)
            maxTotal = max(maxTotal, val);
        
        if (maxTotal < 1.0f) maxTotal = 1.0f; // Minimum scale
        
        // Calculate averages for display
        float avgRender = 0.0f, avgFilm = 0.0f, avgUpload = 0.0f, avgTotal = 0.0f, avgFPS = 0.0f;
        for (size_t i = 0; i < historySize; ++i) {
            avgRender += renderHistory[i];
            avgFilm += filmUpdateHistory[i];
            avgUpload += uploadHistory[i];
            avgTotal += totalHistory[i];
            avgFPS += fpsHistory[i];
        }
        avgRender /= historySize;
        avgFilm /= historySize;
        avgUpload /= historySize;
        avgTotal /= historySize;
        avgFPS /= historySize;
        
        // Create plots with axis and grid using ImPlot-style approach with ImGui
        const ImVec2 graphSize(250, 60);
        const float childHeight = 85.0f;
        
        if (ImGui::BeginChild("RenderGraph", ImVec2(0, childHeight), true)) {
            ImGui::Text("Render Time (ms) - Avg: %.2f", avgRender);
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            ImVec2 plotEnd = ImVec2(plotPos.x + graphSize.x, plotPos.y + graphSize.y);
            
            // Background
            drawList->AddRectFilled(plotPos, plotEnd, IM_COL32(20, 20, 20, 255));
            
            // Grid lines
            for (int i = 0; i <= 4; ++i) {
                float y = plotPos.y + (graphSize.y / 4.0f) * i;
                drawList->AddLine(ImVec2(plotPos.x, y), ImVec2(plotEnd.x, y), IM_COL32(60, 60, 60, 255));
                char label[32];
                snprintf(label, sizeof(label), "%.1f", maxTotal * (1.0f - i / 4.0f));
                drawList->AddText(ImVec2(plotPos.x - 35, y - 7), IM_COL32(180, 180, 180, 255), label);
            }
            
            // Plot data
            for (size_t i = 1; i < historySize; ++i) {
                size_t idx0 = (historyIndex + i - 1) % historySize;
                size_t idx1 = (historyIndex + i) % historySize;
                float x0 = plotPos.x + (float)(i - 1) / (historySize - 1) * graphSize.x;
                float x1 = plotPos.x + (float)i / (historySize - 1) * graphSize.x;
                float y0 = plotEnd.y - (renderHistory[idx0] / maxTotal) * graphSize.y;
                float y1 = plotEnd.y - (renderHistory[idx1] / maxTotal) * graphSize.y;
                drawList->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(100, 200, 255, 255), 2.0f);
            }
            
            drawList->AddRect(plotPos, plotEnd, IM_COL32(100, 100, 100, 255));
            ImGui::Dummy(graphSize);
        }
        ImGui::EndChild();
        
        if (ImGui::BeginChild("FilmGraph", ImVec2(0, childHeight), true)) {
            ImGui::Text("Film Update Time (ms) - Avg: %.2f", avgFilm);
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            ImVec2 plotEnd = ImVec2(plotPos.x + graphSize.x, plotPos.y + graphSize.y);
            
            drawList->AddRectFilled(plotPos, plotEnd, IM_COL32(20, 20, 20, 255));
            
            for (int i = 0; i <= 4; ++i) {
                float y = plotPos.y + (graphSize.y / 4.0f) * i;
                drawList->AddLine(ImVec2(plotPos.x, y), ImVec2(plotEnd.x, y), IM_COL32(60, 60, 60, 255));
                char label[32];
                snprintf(label, sizeof(label), "%.1f", maxTotal * (1.0f - i / 4.0f));
                drawList->AddText(ImVec2(plotPos.x - 35, y - 7), IM_COL32(180, 180, 180, 255), label);
            }
            
            for (size_t i = 1; i < historySize; ++i) {
                size_t idx0 = (historyIndex + i - 1) % historySize;
                size_t idx1 = (historyIndex + i) % historySize;
                float x0 = plotPos.x + (float)(i - 1) / (historySize - 1) * graphSize.x;
                float x1 = plotPos.x + (float)i / (historySize - 1) * graphSize.x;
                float y0 = plotEnd.y - (filmUpdateHistory[idx0] / maxTotal) * graphSize.y;
                float y1 = plotEnd.y - (filmUpdateHistory[idx1] / maxTotal) * graphSize.y;
                drawList->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(100, 255, 100, 255), 2.0f);
            }
            
            drawList->AddRect(plotPos, plotEnd, IM_COL32(100, 100, 100, 255));
            ImGui::Dummy(graphSize);
        }
        ImGui::EndChild();
        
        if (ImGui::BeginChild("UploadGraph", ImVec2(0, childHeight), true)) {
            ImGui::Text("Upload/Draw Time (ms) - Avg: %.2f", avgUpload);
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            ImVec2 plotEnd = ImVec2(plotPos.x + graphSize.x, plotPos.y + graphSize.y);
            
            drawList->AddRectFilled(plotPos, plotEnd, IM_COL32(20, 20, 20, 255));
            
            for (int i = 0; i <= 4; ++i) {
                float y = plotPos.y + (graphSize.y / 4.0f) * i;
                drawList->AddLine(ImVec2(plotPos.x, y), ImVec2(plotEnd.x, y), IM_COL32(60, 60, 60, 255));
                char label[32];
                snprintf(label, sizeof(label), "%.1f", maxTotal * (1.0f - i / 4.0f));
                drawList->AddText(ImVec2(plotPos.x - 35, y - 7), IM_COL32(180, 180, 180, 255), label);
            }
            
            for (size_t i = 1; i < historySize; ++i) {
                size_t idx0 = (historyIndex + i - 1) % historySize;
                size_t idx1 = (historyIndex + i) % historySize;
                float x0 = plotPos.x + (float)(i - 1) / (historySize - 1) * graphSize.x;
                float x1 = plotPos.x + (float)i / (historySize - 1) * graphSize.x;
                float y0 = plotEnd.y - (uploadHistory[idx0] / maxTotal) * graphSize.y;
                float y1 = plotEnd.y - (uploadHistory[idx1] / maxTotal) * graphSize.y;
                drawList->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(255, 200, 100, 255), 2.0f);
            }
            
            drawList->AddRect(plotPos, plotEnd, IM_COL32(100, 100, 100, 255));
            ImGui::Dummy(graphSize);
        }
        ImGui::EndChild();
        
        if (ImGui::BeginChild("TotalGraph", ImVec2(0, childHeight), true)) {
            ImGui::Text("Total Frame Time (ms) - Avg: %.2f", avgTotal);
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            ImVec2 plotEnd = ImVec2(plotPos.x + graphSize.x, plotPos.y + graphSize.y);
            
            drawList->AddRectFilled(plotPos, plotEnd, IM_COL32(20, 20, 20, 255));
            
            for (int i = 0; i <= 4; ++i) {
                float y = plotPos.y + (graphSize.y / 4.0f) * i;
                drawList->AddLine(ImVec2(plotPos.x, y), ImVec2(plotEnd.x, y), IM_COL32(60, 60, 60, 255));
                char label[32];
                snprintf(label, sizeof(label), "%.1f", maxTotal * (1.0f - i / 4.0f));
                drawList->AddText(ImVec2(plotPos.x - 35, y - 7), IM_COL32(180, 180, 180, 255), label);
            }
            
            for (size_t i = 1; i < historySize; ++i) {
                size_t idx0 = (historyIndex + i - 1) % historySize;
                size_t idx1 = (historyIndex + i) % historySize;
                float x0 = plotPos.x + (float)(i - 1) / (historySize - 1) * graphSize.x;
                float x1 = plotPos.x + (float)i / (historySize - 1) * graphSize.x;
                float y0 = plotEnd.y - (totalHistory[idx0] / maxTotal) * graphSize.y;
                float y1 = plotEnd.y - (totalHistory[idx1] / maxTotal) * graphSize.y;
                drawList->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(255, 100, 200, 255), 2.0f);
            }
            
            drawList->AddRect(plotPos, plotEnd, IM_COL32(100, 100, 100, 255));
            ImGui::Dummy(graphSize);
        }
        ImGui::EndChild();
        
        if (ImGui::BeginChild("FPSGraph", ImVec2(0, childHeight), true)) {
            float maxFPS = 0.0f;
            for (float val : fpsHistory)
                maxFPS = max(maxFPS, val);
            if (maxFPS < 10.0f) maxFPS = 10.0f;
            
            ImGui::Text("FPS - Avg: %.1f", avgFPS);
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            ImVec2 plotEnd = ImVec2(plotPos.x + graphSize.x, plotPos.y + graphSize.y);
            
            drawList->AddRectFilled(plotPos, plotEnd, IM_COL32(20, 20, 20, 255));
            
            for (int i = 0; i <= 4; ++i) {
                float y = plotPos.y + (graphSize.y / 4.0f) * i;
                drawList->AddLine(ImVec2(plotPos.x, y), ImVec2(plotEnd.x, y), IM_COL32(60, 60, 60, 255));
                char label[32];
                snprintf(label, sizeof(label), "%.0f", maxFPS * (1.0f - i / 4.0f));
                drawList->AddText(ImVec2(plotPos.x - 35, y - 7), IM_COL32(180, 180, 180, 255), label);
            }
            
            for (size_t i = 1; i < historySize; ++i) {
                size_t idx0 = (historyIndex + i - 1) % historySize;
                size_t idx1 = (historyIndex + i) % historySize;
                float x0 = plotPos.x + (float)(i - 1) / (historySize - 1) * graphSize.x;
                float x1 = plotPos.x + (float)i / (historySize - 1) * graphSize.x;
                float y0 = plotEnd.y - (fpsHistory[idx0] / maxFPS) * graphSize.y;
                float y1 = plotEnd.y - (fpsHistory[idx1] / maxFPS) * graphSize.y;
                drawList->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(255, 255, 100, 255), 2.0f);
            }
            
            drawList->AddRect(plotPos, plotEnd, IM_COL32(100, 100, 100, 255));
            ImGui::Dummy(graphSize);
        }
        ImGui::EndChild();

        ImGui::Separator();
        ImGui::Text("CUDA Status:");
        ImGui::TextColored(cudaOK ? ImVec4(0,1,0,1) : ImVec4(1,0,0,1),
                           cudaOK ? "CUDA Loaded Successfully" : "CUDA Failed");

        ImGui::Separator();
        ImGui::Text("OptiX Status:");
        ImGui::TextColored(optixOK ? ImVec4(0,1,0,1) : ImVec4(1,0,0,1),
                           optixOK ? "OptiX Loaded Successfully" : "OptiX Failed");

        ImGui::Separator();
        ImGui::Text("Renderer");
        for (int i = 0; i < static_cast<int>(rendererOptions.size()); ++i)
        {
            if (ImGui::RadioButton(rendererOptions[i].Label, selectedRenderer == i))
            {
                selectedRenderer = i;
                film.Clear();
            }
        }

        ImGui::End();


        // Render
        ImGui::Render();
        
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0.15f, 0.18f, 0.22f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        Renderer* activeRenderer = nullptr;
        if (!rendererOptions.empty())
        {
            const int clampedIndex = std::clamp(selectedRenderer, 0, static_cast<int>(rendererOptions.size()) - 1);
            activeRenderer = rendererOptions[clampedIndex].Instance.get();
            selectedRenderer = clampedIndex;
        }

        if (activeRenderer)
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            activeRenderer->ProgressiveRender();
            const auto t1 = std::chrono::high_resolution_clock::now();
            if(selectedRenderer == 1 || selectedRenderer == 2)
                film.UpdateDisplayGPU();
            else
                film.UpdateDisplay();
            const auto t2 = std::chrono::high_resolution_clock::now();
            if(selectedRenderer == 1 || selectedRenderer == 2) {
                frame->SetDataGPU(film.GetDisplayDataGPU());
                openglRenderer.Draw(*frame);
            }
            else {
                frame->SetData(film.GetDisplayData());
                openglRenderer.Draw(*frame);
            }
            const auto t3 = std::chrono::high_resolution_clock::now();

            lastRenderMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
            lastFilmUpdateMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
            lastUploadMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
            
            // Accumulate history
            renderHistory[historyIndex] = static_cast<float>(lastRenderMs);
            filmUpdateHistory[historyIndex] = static_cast<float>(lastFilmUpdateMs);
            uploadHistory[historyIndex] = static_cast<float>(lastUploadMs);
            totalHistory[historyIndex] = static_cast<float>(lastRenderMs + lastFilmUpdateMs + lastUploadMs);
            fpsHistory[historyIndex] = static_cast<float>(1.0 / delta);
            historyIndex = (historyIndex + 1) % historySize;
        }
        

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapInterval(0);
        glfwSwapBuffers(window);

        lastTime = currentTime;
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
