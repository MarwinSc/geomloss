#version 460

out vec4 FragColor;
in vec2 TexCoords;

uniform vec3 bg_color;
uniform vec3 depth_contour_color;
uniform vec3 exen_contour_color;
uniform bool color_distance;

uniform sampler2D colorTexture;
uniform sampler2D explicitEncodingTexture;
uniform sampler2D depthTexture;
// depth contour parameters
uniform bool render_depth_contour;
uniform float depth_contour_amp;
// explicit encoding contour parameters
uniform bool render_exen_contour;
uniform float exen_contour_amp;
// general contour parameters
uniform bool contour_overlay;
uniform float offset_v;
uniform float offset_h;
//const float offset_v = 1.0 / (720.0 / 1.0);  
//const float offset_h = 1.0 / (1280.0 / 1.0);

void main()
{
    // kernels and uv offsets
    vec2 offsets_3[9] = vec2[](
        vec2(-offset_h,  offset_v), // top-left
        vec2( 0.0f,    offset_v), // top-center
        vec2( offset_h,  offset_v), // top-right
        vec2(-offset_h,  0.0f),   // center-left
        vec2( 0.0f,    0.0f),   // center-center
        vec2( offset_h,  0.0f),   // center-right
        vec2(-offset_h, -offset_v), // bottom-left
        vec2( 0.0f,   -offset_v), // bottom-center
        vec2( offset_h, -offset_v)  // bottom-right    
    );

    vec2 offsets_5[25] = vec2[](
        vec2(-2*offset_h,  2*offset_v), vec2(-offset_h,  2*offset_v), vec2(0.0,  2*offset_v), vec2(offset_h,  2*offset_v), vec2(2*offset_h,  2*offset_v),
        vec2(-2*offset_h,  offset_v),   vec2(-offset_h,  offset_v),   vec2(0.0,  offset_v),   vec2(offset_h,  offset_v),   vec2(2*offset_h,  offset_v),
        vec2(-2*offset_h,  0.0),      vec2(-offset_h,  0.0),      vec2(0.0,  0.0),      vec2(offset_h,  0.0),      vec2(2*offset_h,  0.0),
        vec2(-2*offset_h, -offset_v),   vec2(-offset_h, -offset_v),   vec2(0.0, -offset_v),   vec2(offset_h, -offset_v),   vec2(2*offset_h, -offset_v),
        vec2(-2*offset_h, -2*offset_v), vec2(-offset_h, -2*offset_v), vec2(0.0, -2*offset_v), vec2(offset_h, -2*offset_v), vec2(2*offset_h, -2*offset_v)
    );

    float kernel3[9] = float[](
        1, 1, 1,
        1,-8, 1,
        1, 1, 1
    );

    float kernel5[25] = float[](
        1,  1,  1,  1,  1,
        1,  2,  2,  2,  1,
        1,  2, -32, 2,  1,
        1,  2,  2,  2,  1,
        1,  1,  1,  1,  1
    );
    
    float depth_contour = 0.0;
    float exen_contour = 0.0;
    
    // render the depth contour
    if (render_depth_contour)
    {
        // sample the 3x3 kernel centered around the current pixel
        float sampleTex[25];
        for(int i = 0; i < 25; i++)
        {
            sampleTex[i] = float(texture(depthTexture, TexCoords.st + (offsets_5[i])));
        }
        float col = float(0.0);
        for(int i = 0; i < 25; i++)
            col += sampleTex[i] * kernel5[i];

        // add user controllable factor to adjust the effect
        depth_contour = clamp(col * depth_contour_amp, 0.0, 1.0);
    }
    
    if (render_exen_contour){
        // sample the 3x3 kernel centered around the current pixel
        float sampleTex[25];
        for(int i = 0; i < 25; i++)
        {
            sampleTex[i] = float(texture(explicitEncodingTexture, TexCoords.st + (offsets_5[i])));
        }
        float col = float(0.0);
        for(int i = 0; i < 25; i++)
            col += sampleTex[i] * kernel5[i];

        // add user controllable factor to adjust the effect
        exen_contour = clamp(col * exen_contour_amp, 0.0, 1.0);
    }

    vec3 color = vec3(1.0);

    // Compose final color
    if (contour_overlay){
        if (color_distance) {
            color = vec3(texture(explicitEncodingTexture, TexCoords).xyz);
        } else {
            color = vec3(texture(colorTexture, TexCoords).xyz);
        }
    // draw the contours alone
    }else{
        color = bg_color;
    }

    if ((depth_contour > 0.0) || (exen_contour > 0.0)){
        vec3 blend_contours = clamp(depth_contour * (1 - depth_contour_color) + exen_contour * (1 - exen_contour_color), 0.0, 1.0);
        color = clamp(color - blend_contours, 0.0, 1.0);
    } 
    FragColor = vec4(color, 1.0);
}  