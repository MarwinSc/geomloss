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
uniform float exen_number_contour_lines;
// general contour parameters
uniform bool contour_overlay;
uniform float offset_v;
uniform float offset_h;
//const float offset_v = 1.0 / (720.0 / 1.0);  
//const float offset_h = 1.0 / (1280.0 / 1.0);

const int KERNEL_SIZE = 5;
const int TOTAL_KERNEL_SIZE = KERNEL_SIZE * KERNEL_SIZE;

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

    float kernel3[9] = float[](
        1, 1, 1,
        1,-8, 1,
        1, 1, 1
    );

    vec2 offsets[25] = vec2[](
        vec2(-2*offset_h,  2*offset_v), vec2(-offset_h,  2*offset_v), vec2(0.0,  2*offset_v), vec2(offset_h,  2*offset_v), vec2(2*offset_h,  2*offset_v),
        vec2(-2*offset_h,  offset_v),   vec2(-offset_h,  offset_v),   vec2(0.0,  offset_v),   vec2(offset_h,  offset_v),   vec2(2*offset_h,  offset_v),
        vec2(-2*offset_h,  0.0),      vec2(-offset_h,  0.0),      vec2(0.0,  0.0),      vec2(offset_h,  0.0),      vec2(2*offset_h,  0.0),
        vec2(-2*offset_h, -offset_v),   vec2(-offset_h, -offset_v),   vec2(0.0, -offset_v),   vec2(offset_h, -offset_v),   vec2(2*offset_h, -offset_v),
        vec2(-2*offset_h, -2*offset_v), vec2(-offset_h, -2*offset_v), vec2(0.0, -2*offset_v), vec2(offset_h, -2*offset_v), vec2(2*offset_h, -2*offset_v)
    );

    float kernel[25] = float[](
        1,  1,  1,  1,  1,
        1,  2,  2,  2,  1,
        1,  2, -32, 2,  1,
        1,  2,  2,  2,  1,
        1,  1,  1,  1,  1
    );

    //initKernel();
    //initOffsets(offset_h, offset_v);
    
    float depth_contour = 0.0;
    float exen_contour = 0.0;

    // render the depth contour
    if (render_depth_contour)
    {
        // sample the 3x3 kernel centered around the current pixel
        float sampleTex[TOTAL_KERNEL_SIZE];
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
        {
            sampleTex[i] = float(texture(depthTexture, TexCoords.st + (offsets[i])));
        }
        float col = float(0.0);
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
            col += sampleTex[i] * kernel[i];

        // add user controllable factor to adjust the effect
        depth_contour = clamp(col * depth_contour_amp, 0.0, 1.0);
    }
    
    if (render_exen_contour){
        // sample the kernel centered around the current pixel
        float sampleTex[TOTAL_KERNEL_SIZE];
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
        {   
            vec4 tex_color = texture(explicitEncodingTexture, TexCoords.st + (offsets[i]));
            // binary thresholding
            float weight = tex_color.r;
            // Scale the value into the range [0, 4] (5 discrete steps: 0.0, 0.25, 0.5, 0.75, 1.0)
            float scaledValue = weight * exen_number_contour_lines;
            // Round the scaled value to the nearest integer (this maps it to [0, 1, 2, 3, 4])
            float rounded = round(scaledValue);
            // Map back to the closest value in the set [0.0, 0.25, 0.5, 0.75, 1.0]
            sampleTex[i] = rounded * (1.0/exen_number_contour_lines);
        }
        float col = float(0.0);
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
            col += sampleTex[i] * kernel[i];

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
        float weight = (depth_contour > 0.0 && exen_contour > 0.0) ? 0.5 : 
                    (depth_contour > 0.0 || exen_contour > 0.0) ? 1.0 : 0.0;

        color = mix(color, clamp((depth_contour * depth_contour_color) + (exen_contour * exen_contour_color), 0.0, 1.0), (depth_contour + exen_contour) * weight);
    }

    FragColor = vec4(color, 1.0);
}  