
#version 460

uniform float transparency;

in vec4 color; // w is the distance in the range[0, 1]
out vec4 outColor;
out vec4 outExplicitEncoding;

void main() {
    // Calculate the distance from the center of the point
    // gl_PointCoord is available when redering points. It's basically an uv coordinate.
    //float dist = step(length(gl_PointCoord.xy - vec2(0.5)), 0.5);

    // .. an use to render a circle!
    //outColor = vec4(dist * color, dist);

    outColor = vec4(color.xyz, transparency);

    float interp = color.w;
    outExplicitEncoding = vec4(0.0, 0.0, 1.0, 1.0) * (1 - interp) + vec4(1.0, 0.0, 0.0, 1.0) * interp;
    outExplicitEncoding.w = transparency;
}