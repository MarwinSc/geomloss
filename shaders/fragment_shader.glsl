
#version 460

in vec4 color;
out vec4 outColor;

void main() {
    // Calculate the distance from the center of the point
    // gl_PointCoord is available when redering points. It's basically an uv coordinate.
    //float dist = step(length(gl_PointCoord.xy - vec2(0.5)), 0.5);

    // .. an use to render a circle!
    //outColor = vec4(dist * color, dist);

    outColor = color;
}