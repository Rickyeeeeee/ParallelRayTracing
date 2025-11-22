#pragma once

#include "opengl_utils.h"   // OpenGLTexture, OpenGLProgram
#include <glad/glad.h>

/**
 * Simple helper to render a 2D texture to the current framebuffer
 * using a fullscreen triangle/quad.
 */
class OpenGLTextureRenderer
{
public:
    // vertex/fragment shaders should have:
    // layout(location = 0) in vec2 aPos;
    // layout(location = 1) in vec2 aTexCoord;
    // uniform sampler2D uTexture;
    OpenGLTextureRenderer();
    ~OpenGLTextureRenderer();

    // Draw given texture to current framebuffer
    void Draw(const OpenGLTexture& texture);

private:
    void InitFullscreenQuad();

private:
    OpenGLProgram m_Program;
    GLuint m_VAO = 0;
    GLuint m_VBO = 0;
    GLint  m_TextureUniformLocation = -1;
};
