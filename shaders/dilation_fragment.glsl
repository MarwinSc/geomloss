#version 460 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D edgeTexture;
uniform vec3 parameters;

void main() {
    int   size         = int(parameters.x);
    float offset_h     =     parameters.y;
    float offset_v     =     parameters.z;

    //vec2 texSize   = textureSize(edgeTexture, 0).xy;
    //vec2 fragCoord = gl_FragCoord.xy;

    vec4 cc = texture(edgeTexture, TexCoords.st);

    if (size <= 0) { return; }

    float max_edge = cc.a;
    vec4 color_max_edge = cc;

    for (int i = -size; i <= size; i++) {
        for (int j = -size; j <= size; j++) {
            // For a rectangular shape.
            //if (false);

            // For a diamond shape;
            //if (!(abs(i) <= size - abs(j))) { continue; }

            // For a circular shape.
            //if (!(distance(vec2(j, i), vec2(0, 0)) <= size)) { continue; }

            vec2 offset = vec2(i * offset_h, j * offset_v);
            vec4 c = texture(edgeTexture, TexCoords.st + offset);

            if (c.a > max_edge) {
                max_edge = c.a;
                color_max_edge = c;
            }
        }
    }

    FragColor = color_max_edge;
}