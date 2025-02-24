
#version 460

in vec4 in_position; // w is the source distance
in vec4 in_color; // w is the target distance

uniform mat4 projection;
uniform mat4 modelview;
uniform float point_size;
//uniform float time;
uniform bool varying_size;
uniform float color_state;

uniform bool constant_color;
uniform float filter_treshold;
uniform float transparency;

out vec4 color;
out vec4 xen_color;

void main() {

    // set vertex_position.w to 1.0
    gl_Position = projection * modelview * vec4(in_position.xyz, 1.0);

    // Set the point size
    if (varying_size) {
        gl_PointSize = min((1/gl_Position.z) * point_size, 20.0);
    } else {
        gl_PointSize = point_size;
    }

    color = vec4(in_color.xyz, 1.0);

    // Set the explicit encoding colors
    // interpolate between previous and current exen color
    
    float interp = mix(in_position.w, in_color.w, color_state);
    if (constant_color) {
        interp = in_color.w;
    }
    xen_color = vec4(0.0, 0.0, 1.0, 1.0) * (1 - interp) + vec4(1.0, 0.0, 0.0, 1.0) * interp;
    // if below filter threshold don't render
    if (interp < filter_treshold) {
        xen_color.w = transparency; 
        if (transparency < 0.01) {
            gl_Position = vec4(-2, -2, 0, 0);
        }
    } 
}   