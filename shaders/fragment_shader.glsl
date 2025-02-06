
#version 460

uniform float transparency;

in vec4 color; // w is the distance in the range[0, 1]
out vec4 outColor;
out vec4 outExplicitEncoding;

void main() {

    // Convert gl_PointCoord to range [-0.5, 0.5]
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord) * 2.0;

    if (dist > 1.0) discard;

    // color with transparency
    outColor = vec4(color.xyz, transparency);

    float interp = color.w;
    outExplicitEncoding = vec4(0.0, 0.0, 1.0, 1.0) * (1 - interp) + vec4(1.0, 0.0, 0.0, 1.0) * interp;
    outExplicitEncoding.w = transparency;
}