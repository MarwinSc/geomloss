#version 460

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D colorTexture;
uniform sampler2D depthEdgesTexture;
uniform sampler2D exenEdgesTexture;

uniform bool contour_overlay;

void main()
{
    vec4 color = texture(colorTexture, TexCoords);
    vec4 depth_edges = texture(depthEdgesTexture, TexCoords);
    vec4 exen_edges = texture(exenEdgesTexture, TexCoords);

    vec4 edges = clamp(depth_edges + exen_edges, 0.0, 1.0);

    if (contour_overlay){
        FragColor = vec4(mix(color.xyz, edges.xyz, edges.w), 1.0);
    }
    else{
        //FragColor = vec4(vec3(edges.w), 1.0);
        FragColor = edges;
    }
}   