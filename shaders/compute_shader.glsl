
#version 460
#define GROUP_SIZE %COMPUTE_SIZE%

layout(local_size_x=GROUP_SIZE) in;

//uniform float time;
uniform float transition_state;
uniform float color_state;
//uniform float max_distance;
uniform bool color_distance;

struct Point{
    vec4 pos; // for the assignment w is the distance
    vec4 col;
};

layout(std430, binding=0) buffer source{
    Point points[];
} In;
layout(std430, binding=1) buffer points_out{
    Point points[];
} Out;
layout(std430, binding=2) buffer target{
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
    out_point.pos.xyz = src_pos.xyz * (transition_state) + tar_pos.xyz * (1 - transition_state);
    out_point.pos.w = src_pt.pos.w;

    if(color_distance){
        //float d = distance(p.xyz, target_p.xyz);
        //float interp = ((1 - (d / Ass.assignments[x].pos.w)) * (Ass.assignments[x].pos.w / max_distance));
        float interp = Ass.points[x].pos.w;

        out_point.col.xyzw = vec4(0.0, 0.0, 1.0, 1.0) * (1 - interp) + vec4(1.0, 0.0, 0.0, 1.0) * interp;

        out_point.col.w = interp;
    }else{
        out_point.col.xyzw = src_pt.col.xyzw * color_state + Ass.points[x].col.xyzw * (1 - color_state);
        //out_point.col.xyzw = in_point.col.xyzw;
    }
    Out.points[x] = out_point;
}