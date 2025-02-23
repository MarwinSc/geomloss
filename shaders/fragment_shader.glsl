
#version 460

in vec4 color;
in vec4 xen_color;

out vec4 outColor;
out vec4 outExplicitEncoding;

void main() {

    // Convert gl_PointCoord to range [-0.5, 0.5]
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord) * 2.0;

    if (dist > 1.0) discard;

    // color with transparency
    outColor = color;
    outExplicitEncoding = xen_color;
    
}