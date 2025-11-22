#include "opengl_renderer.h"
#include <core/core.h>

OpenGLTextureRenderer::OpenGLTextureRenderer()
    : m_Program((assetRoot + "/shaders/quad_vert.glsl").c_str(), 
                (assetRoot + "/shaders/quad_frag.glsl").c_str())
{
    InitFullscreenQuad();

    // Cache uniform location for sampler2D
    m_TextureUniformLocation =
        glGetUniformLocation(m_Program.GetShaderID(), "uTexture");
}

OpenGLTextureRenderer::~OpenGLTextureRenderer()
{
    if (m_VBO)
        glDeleteBuffers(1, &m_VBO);
    if (m_VAO)
        glDeleteVertexArrays(1, &m_VAO);
}

void OpenGLTextureRenderer::InitFullscreenQuad()
{
    // Fullscreen quad: positions in clip space, texcoords in [0,1]
    float quadVertices[] = {
        //  pos      // texcoord
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,

        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };

    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices),
                 quadVertices, GL_STATIC_DRAW);

    // layout(location = 0) vec2 aPos;
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE,
        4 * sizeof(float),
        (void*)0
    );

    // layout(location = 1) vec2 aTexCoord;
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE,
        4 * sizeof(float),
        (void*)(2 * sizeof(float))
    );

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void OpenGLTextureRenderer::Draw(const OpenGLTexture& texture)
{
    // If you want depth-tested compositing, remove this line
    glDisable(GL_DEPTH_TEST);

    glUseProgram(m_Program.GetShaderID());

    // Bind texture to unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture.GetTextureID());

    if (m_TextureUniformLocation >= 0)
        glUniform1i(m_TextureUniformLocation, 0);

    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}
