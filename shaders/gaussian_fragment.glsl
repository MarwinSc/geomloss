#version 460

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D Texture;

uniform float offset_h;
uniform float offset_v;

const float kernel_5[25] = float[](
    1,    4,    6,    4,    1,
    4,   16,   24,   16,    4,
    6,   24,   36,   24,    6,
    4,   16,   24,   16,    4,
    1,    4,    6,    4,    1
);

void main()
{
    vec4 blurredColor = vec4(0.0);
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 offset = vec2(j, -i) * vec2(offset_h, offset_v);
            blurredColor += texture(Texture, TexCoords + offset) * ((1.0/256.0) * kernel_5[(i+2) * (j+2) + (j+2)]);
        }
    }

    blurredColor = clamp(blurredColor * 2.0, 0.0, 1.0);

    FragColor = blurredColor;
}  