#version 450 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTexture;

void main()
{
    // Flip V so top-left film origin matches GL texture space
    FragColor = texture(uTexture, vec2(vTexCoord.x, 1.0 - vTexCoord.y));
}
