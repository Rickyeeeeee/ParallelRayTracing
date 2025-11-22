#pragma once
#include <Core/Core.h>
#include <glad/glad.h>

#include <memory>

class OpenGLTexture
{
public:
    OpenGLTexture(uint32_t width, uint32_t height);
    ~OpenGLTexture();


    void SetData(const void* data);


    uint32_t GetWidth() const { return m_Width; }
    uint32_t GetHeight() const { return m_Height; }
    uint32_t GetTextureID() const { return m_TextureID; }


private:
    uint32_t m_Width;
    uint32_t m_Height;
    uint32_t m_TextureID;
};


class OpenGLProgram
{
public:
    OpenGLProgram(const char* vertexPath, const char* fragmentPath);
    ~OpenGLProgram();

    uint32_t GetShaderID() const { return m_ProgramID; }

private:
    uint32_t m_ProgramID;
    uint32_t m_VertexShaderID;
    uint32_t m_FragmentShaderID;
};

class OpenGLFramebuffer
{
public:
    OpenGLFramebuffer(uint32_t width, uint32_t height);
    ~OpenGLFramebuffer();


    void Bind() const;
    void Unbind() const;


    uint32_t GetColorAttachment() const { return m_ColorAttachment; }
    uint32_t GetDepthAttachment() const { return m_DepthAttachment; }
    uint32_t GetID() const { return m_FBO; }

private:
    uint32_t m_Width;
    uint32_t m_Height;


    uint32_t m_FBO;
    uint32_t m_ColorAttachment;
    uint32_t m_DepthAttachment;
};