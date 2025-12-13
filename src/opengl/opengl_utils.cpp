#include "opengl_utils.h"
#include <glad/glad.h>
#include <fstream>
#include <sstream>
#include <string>

OpenGLTexture::OpenGLTexture(uint32_t width, uint32_t height)
    : m_Width(width), m_Height(height)
{
    glGenTextures(1, &m_TextureID);
    glBindTexture(GL_TEXTURE_2D, m_TextureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 static_cast<GLsizei>(m_Width),
                 static_cast<GLsizei>(m_Height),
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);
}

OpenGLTexture::~OpenGLTexture()
{
    if (m_TextureID)
        glDeleteTextures(1, &m_TextureID);
}

void OpenGLTexture::SetData(const void* data)
{
    glBindTexture(GL_TEXTURE_2D, m_TextureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    static_cast<GLsizei>(m_Width),
                    static_cast<GLsizei>(m_Height),
                    GL_RGBA, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGLTexture::SetDataGPU(const void* d_data)
{
    glBindTexture(GL_TEXTURE_2D, m_TextureID);
    // TODO: find an API to upload data from GPU memory directly
    glBindTexture(GL_TEXTURE_2D, 0);
}

// Simple helper to read an entire text file into a std::string
static std::string ReadFileToString(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return {}; // empty on failure

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

OpenGLProgram::OpenGLProgram(const char* vertexPath, const char* fragmentPath)
    : m_ProgramID(0), m_VertexShaderID(0), m_FragmentShaderID(0)
{
    // --- Vertex shader ---
    m_VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    std::string vertexSource = ReadFileToString(vertexPath);
    const char* vSrc = vertexSource.c_str();
    glShaderSource(m_VertexShaderID, 1, &vSrc, nullptr);
    glCompileShader(m_VertexShaderID);
    // (minimal: no error checking to keep it short)

    // --- Fragment shader ---
    m_FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
    std::string fragmentSource = ReadFileToString(fragmentPath);
    const char* fSrc = fragmentSource.c_str();
    glShaderSource(m_FragmentShaderID, 1, &fSrc, nullptr);
    glCompileShader(m_FragmentShaderID);
    // (minimal: no error checking)

    // --- Link program ---
    m_ProgramID = glCreateProgram();
    glAttachShader(m_ProgramID, m_VertexShaderID);
    glAttachShader(m_ProgramID, m_FragmentShaderID);
    glLinkProgram(m_ProgramID);
    // (minimal: no error checking)
}

OpenGLProgram::~OpenGLProgram()
{
    if (m_ProgramID)
        glDeleteProgram(m_ProgramID);
    if (m_VertexShaderID)
        glDeleteShader(m_VertexShaderID);
    if (m_FragmentShaderID)
        glDeleteShader(m_FragmentShaderID);
}


OpenGLFramebuffer::OpenGLFramebuffer(uint32_t width, uint32_t height)
    : m_Width(width), m_Height(height)
{
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

    glGenTextures(1, &m_ColorAttachment);
    glBindTexture(GL_TEXTURE_2D, m_ColorAttachment);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 (GLsizei)m_Width, (GLsizei)m_Height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, m_ColorAttachment, 0);

    glGenRenderbuffers(1, &m_DepthAttachment);
    glBindRenderbuffer(GL_RENDERBUFFER, m_DepthAttachment);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                          (GLsizei)m_Width, (GLsizei)m_Height);

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, m_DepthAttachment);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

OpenGLFramebuffer::~OpenGLFramebuffer()
{
    if (m_DepthAttachment)
        glDeleteRenderbuffers(1, &m_DepthAttachment);
    if (m_ColorAttachment)
        glDeleteTextures(1, &m_ColorAttachment);
    if (m_FBO)
        glDeleteFramebuffers(1, &m_FBO);
}

void OpenGLFramebuffer::Bind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, (GLsizei)m_Width, (GLsizei)m_Height);
}

void OpenGLFramebuffer::Unbind() const
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
