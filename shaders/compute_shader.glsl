
#version 460
#define GROUP_SIZE %COMPUTE_SIZE%

layout(local_size_x=GROUP_SIZE) in;

//uniform float time;
uniform float transition_state;
uniform float color_state;

struct Point{
    vec4 pos; 
    vec4 col;
};

layout(std430, binding=0) buffer source{ // col.w is the distance in the range [0, 1] to the reference
    Point points[];
} In;
layout(std430, binding=1) buffer points_out{ // pos.w is the source distance, col.w is the target distance
    Point points[];
} Out;
layout(std430, binding=2) buffer target{ // col.w is the distance in the range [0, 1] to the reference
     Point points[];
} Ass;

void main()
{
    int x = int(gl_GlobalInvocationID);
    if(In.points.length() <= x){
        return;
    }

    Point src_pt = In.points[x];
    vec4 src_pos = src_pt.pos.xyzw;
    vec3 tar_pos = Ass.points[x].pos.xyz;
 
    Point out_point;
    out_point.pos.xyz = src_pos.xyz * (1 - transition_state) + tar_pos.xyz * (transition_state);
    out_point.pos.w = src_pt.col.w; 
    
    out_point.col.xyzw = src_pt.col.xyzw * (1 - color_state) + Ass.points[x].col.xyzw * (color_state);
    out_point.col.w = Ass.points[x].col.w;

    Out.points[x] = out_point;
}