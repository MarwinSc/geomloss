#version 460

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D colorTexture;
uniform sampler2D edgesTexture;

uniform bool contour_overlay;

void main()
{
    vec4 color = texture(colorTexture, TexCoords);
    vec4 edges = texture(edgesTexture, TexCoords);

    if (contour_overlay){
        FragColor = vec4(mix(color.xyz, edges.xyz, edges.w), 1.0);
    }
    else{
        //FragColor = vec4(vec3(edges.w), 1.0);
        FragColor = edges;
    }
}   