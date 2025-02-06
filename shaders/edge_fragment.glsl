#version 460

out vec4 FragColor;
in vec2 TexCoords;

// uniform sampler2D colorTexture;
uniform sampler2D explicitEncodingTexture;
uniform sampler2D depthTexture;

// uniform vec3 bg_color;
uniform vec3 depth_contour_color;
uniform vec3 exen_contour_color;

// depth contour parameters
uniform bool render_depth_contour;
uniform float depth_contour_amp;

// explicit encoding contour parameters
uniform bool render_exen_contour;
uniform float exen_contour_amp;
uniform float exen_number_contour_lines;

// general contour parameters
uniform float offset_v;
uniform float offset_h;

const int KERNEL_SIZE = 5;
const int TOTAL_KERNEL_SIZE = KERNEL_SIZE * KERNEL_SIZE;

void main()
{

    // kernels and uv offsets
    const vec2 offsets_3[9] = vec2[](
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

    const float kernel3[9] = float[](
        1, 1, 1,
        1,-8, 1,
        1, 1, 1
    );

    const vec2 offsets_5[25] = vec2[](
        vec2(-2*offset_h,  2*offset_v), vec2(-offset_h,  2*offset_v), vec2(0.0,  2*offset_v), vec2(offset_h,  2*offset_v), vec2(2*offset_h,  2*offset_v),
        vec2(-2*offset_h,  offset_v),   vec2(-offset_h,  offset_v),   vec2(0.0,  offset_v),   vec2(offset_h,  offset_v),   vec2(2*offset_h,  offset_v),
        vec2(-2*offset_h,  0.0),      vec2(-offset_h,  0.0),      vec2(0.0,  0.0),      vec2(offset_h,  0.0),      vec2(2*offset_h,  0.0),
        vec2(-2*offset_h, -offset_v),   vec2(-offset_h, -offset_v),   vec2(0.0, -offset_v),   vec2(offset_h, -offset_v),   vec2(2*offset_h, -offset_v),
        vec2(-2*offset_h, -2*offset_v), vec2(-offset_h, -2*offset_v), vec2(0.0, -2*offset_v), vec2(offset_h, -2*offset_v), vec2(2*offset_h, -2*offset_v)
    );

    const float kernel_5[25] = float[](
        1,  1,  1,  1,  1,
        1,  2,  2,  2,  1,
        1,  2, -32, 2,  1,
        1,  2,  2,  2,  1,
        1,  1,  1,  1,  1
    );

    const float neighbour_kernel_5[25] = float[](
        1,  1,  1,  1,  1,
        1,  0,  0,  0,  1,
        1,  0,  0,  0,  1,
        1,  0,  0,  0,  1,
        1,  1,  1,  1,  1
    );

    const float kernel_9[81] = float[](
        0,  1,  1,  2,  2,  2,  1,  1,  0,
        1,  2,  4,  5,  5,  5,  4,  2,  1,
        1,  4,  5,  3,  0,  3,  5,  4,  1,
        2,  5,  3,-12,-24,-12,  3,  5,  2,
        2,  5,  0,-24,-40,-24,  0,  5,  2,
        2,  5,  3,-12,-24,-12,  3,  5,  2,
        1,  4,  5,  3,  0,  3,  5,  4,  1,
        1,  2,  4,  5,  5,  5,  4,  2,  1,
        0,  1,  1,  2,  2,  2,  1,  1,  0
    );

    const float neighbour_kernel_9[81] = float[](
        1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1
    );

    const vec2 offsets_9[81] = vec2[](
        vec2(-4*offset_h,  4*offset_v), vec2(-3*offset_h,  4*offset_v), vec2(-2*offset_h,  4*offset_v), vec2(-offset_h,  4*offset_v), vec2(0.0,  4*offset_v), vec2(offset_h,  4*offset_v), vec2(2*offset_h,  4*offset_v), vec2(3*offset_h,  4*offset_v), vec2(4*offset_h,  4*offset_v),
        vec2(-4*offset_h,  3*offset_v), vec2(-3*offset_h,  3*offset_v), vec2(-2*offset_h,  3*offset_v), vec2(-offset_h,  3*offset_v), vec2(0.0,  3*offset_v), vec2(offset_h,  3*offset_v), vec2(2*offset_h,  3*offset_v), vec2(3*offset_h,  3*offset_v), vec2(4*offset_h,  3*offset_v),
        vec2(-4*offset_h,  2*offset_v), vec2(-3*offset_h,  2*offset_v), vec2(-2*offset_h,  2*offset_v), vec2(-offset_h,  2*offset_v), vec2(0.0,  2*offset_v), vec2(offset_h,  2*offset_v), vec2(2*offset_h,  2*offset_v), vec2(3*offset_h,  2*offset_v), vec2(4*offset_h,  2*offset_v),
        vec2(-4*offset_h,  offset_v),   vec2(-3*offset_h,  offset_v),   vec2(-2*offset_h,  offset_v),   vec2(-offset_h,  offset_v),   vec2(0.0,  offset_v),   vec2(offset_h,  offset_v),   vec2(2*offset_h,  offset_v),   vec2(3*offset_h,  offset_v),   vec2(4*offset_h,  offset_v),
        vec2(-4*offset_h,  0.0),        vec2(-3*offset_h,  0.0),        vec2(-2*offset_h,  0.0),        vec2(-offset_h,  0.0),        vec2(0.0,  0.0),        vec2(offset_h,  0.0),        vec2(2*offset_h,  0.0),        vec2(3*offset_h,  0.0),        vec2(4*offset_h,  0.0),
        vec2(-4*offset_h, -offset_v),   vec2(-3*offset_h, -offset_v),   vec2(-2*offset_h, -offset_v),   vec2(-offset_h, -offset_v),   vec2(0.0, -offset_v),   vec2(offset_h, -offset_v),   vec2(2*offset_h, -offset_v),   vec2(3*offset_h, -offset_v),   vec2(4*offset_h, -offset_v),
        vec2(-4*offset_h, -2*offset_v), vec2(-3*offset_h, -2*offset_v), vec2(-2*offset_h, -2*offset_v), vec2(-offset_h, -2*offset_v), vec2(0.0, -2*offset_v), vec2(offset_h, -2*offset_v), vec2(2*offset_h, -2*offset_v), vec2(3*offset_h, -2*offset_v), vec2(4*offset_h, -2*offset_v),
        vec2(-4*offset_h, -3*offset_v), vec2(-3*offset_h, -3*offset_v), vec2(-2*offset_h, -3*offset_v), vec2(-offset_h, -3*offset_v), vec2(0.0, -3*offset_v), vec2(offset_h, -3*offset_v), vec2(2*offset_h, -3*offset_v), vec2(3*offset_h, -3*offset_v), vec2(4*offset_h, -3*offset_v),
        vec2(-4*offset_h, -4*offset_v), vec2(-3*offset_h, -4*offset_v), vec2(-2*offset_h, -4*offset_v), vec2(-offset_h, -4*offset_v), vec2(0.0, -4*offset_v), vec2(offset_h, -4*offset_v), vec2(2*offset_h, -4*offset_v), vec2(3*offset_h, -4*offset_v), vec2(4*offset_h, -4*offset_v)
    );

    float depth_contour = 0.0;
    float exen_contour = 0.0;

    // todo only one loop
    
    // render the depth contour
    if (render_depth_contour)
    {
        // sample the 5x5 kernel centered around the current pixel
        float sampleTex[TOTAL_KERNEL_SIZE];
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
        {
            sampleTex[i] = float(texture(depthTexture, TexCoords.st + (offsets_5[i])));
        }
        float col = float(0.0);
        // sum to see if it is an edge or rather a individual point
        float sum = float(0.0);
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++){
            col += sampleTex[i] * kernel_5[i];
            sum += sampleTex[i] * neighbour_kernel_5[i];
        }
        if (sum == 0.0){
            depth_contour = 0.0;
        }else{
            // add user controllable factor to adjust the effect
            depth_contour = clamp(col * depth_contour_amp, 0.0, 1.0);
        }
    }
    
    if (render_exen_contour){
        // sample the kernel centered around the current pixel
        float sampleTex[TOTAL_KERNEL_SIZE];
        for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
        {   
            vec4 tex_color = texture(explicitEncodingTexture, TexCoords.st + (offsets_5[i]));
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
            col += sampleTex[i] * kernel_5[i];

        // add user controllable factor to adjust the effect
        exen_contour = clamp(col * exen_contour_amp, 0.0, 1.0);
    }

    if ((depth_contour > 0.01) || (exen_contour > 0.01)){

        float weight = 1.0;
        if (depth_contour > 0.01 && exen_contour > 0.01){
            weight = 0.5;   
        }

        //color = mix(color, clamp((depth_contour * depth_contour_color) + (exen_contour * exen_contour_color), 0.0, 1.0), (depth_contour + exen_contour) * weight);
        vec3 color = clamp((depth_contour * depth_contour_color) + (exen_contour * exen_contour_color), 0.0, 1.0);
        FragColor = vec4(color, (depth_contour + exen_contour) * weight);
    }else{
        FragColor = vec4(1.0, 1.0, 1.0, 0.0);
    }

}  