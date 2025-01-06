
#version 460

in vec4 in_position;
in vec4 in_color;

uniform mat4 projection;
uniform mat4 modelview;
uniform float point_size;
//uniform float time;
uniform bool varying_size;

out vec4 color;

void main() {
    gl_Position = projection * modelview * in_position;
    vec4 position_camera_coord = modelview * in_position;
    // Set the point size
    //gl_PointSize = 25 - gl_Position.z + sin((time + gl_VertexID) * 7.0) * 10.0;

    if (varying_size) {
        gl_PointSize = in_color.w * point_size;
    } else {
        gl_PointSize = point_size;
    }

    // Calculate a random color based on the vertex index
    //color = vec3(mod(gl_VertexID * 432.43, 1.0), mod(gl_VertexID * 6654.32, 1.0), mod(gl_VertexID  * 6544.11, 1.0));
    color = vec4(in_color.xyz, 1.0);
}