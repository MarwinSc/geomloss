#version 460

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D colorTexture;
uniform sampler2D exenTexture;
uniform sampler2D depthEdgesTexture;
uniform sampler2D exenEdgesTexture;

uniform bool colors_and_contour;
uniform bool exen_overlay;
uniform float lower_filter_treshold;
uniform float upper_filter_treshold;

void main()
{
    vec4 color = texture(colorTexture, TexCoords);
    vec4 exen = texture(exenTexture, TexCoords);
    vec4 depth_edges = texture(depthEdgesTexture, TexCoords);
    vec4 exen_edges = texture(exenEdgesTexture, TexCoords);

    vec4 edges = clamp(depth_edges + exen_edges, 0.0, 1.0);

    if (colors_and_contour){
        FragColor = vec4(mix(color.rgb, edges.rgb, edges.a), 1.0);
        if (exen_overlay){
            // todo only using the red channels isn't a good solution 
            // would need to check if the value is above/below the threshold wrt the colormap 
            if (exen.r < lower_filter_treshold || upper_filter_treshold < exen.r){
                // render explicit encoding overlay
                FragColor = vec4(mix(exen.rgb, edges.rgb, edges.a), 1.0);
            }
        }
    }
    else{
        //FragColor = vec4(vec3(edges.w), 1.0);
        FragColor = edges;
    }
}   