#version 460

out vec4 FragColor;
in vec2 TexCoords;

// uniform sampler2D colorTexture;
uniform sampler2D explicitEncodingTexture;

uniform vec3 exen_contour_color;

// explicit encoding contour parameters
uniform float exen_contour_amp;
uniform float exen_number_contour_lines;

// general contour parameters
uniform float offset_v;
uniform float offset_h;
uniform bool opaque;

const int KERNEL_SIZE = 3;
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

    const float kernel_3[9] = float[](
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

    float exen_contour = 0.0;

    // sample the kernel centered around the current pixel
    float sampleTex[TOTAL_KERNEL_SIZE];
    for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
    {   
        vec4 tex_color = texture(explicitEncodingTexture, TexCoords.st + (offsets_3[i]));
        // binary thresholding
        float weight = tex_color.r;
        // Scale the value into the range [0, 4] (5 discrete steps: 0.0, 0.25, 0.5, 0.75, 1.0)
        float scaledValue = weight * exen_number_contour_lines;
        // Round the scaled value to the nearest integer (this maps it to [0, 1, 2, 3, 4])
        float rounded = floor(scaledValue + 0.5);
        // Map back to the closest value in the set [0.0, 0.25, 0.5, 0.75, 1.0]
        sampleTex[i] = rounded * (1.0/exen_number_contour_lines);
    }

    float col = float(0.0);
    for(int i = 0; i < TOTAL_KERNEL_SIZE; i++)
        col += sampleTex[i] * kernel_3[i];

    // add user controllable factor to adjust the effect
    exen_contour = clamp(col * exen_contour_amp, 0.0, 1.0);

    if (exen_contour > 0.0){

        vec3 color = clamp(exen_contour * exen_contour_color, 0.0, 1.0);

        if (opaque)
            FragColor = vec4(color, 1.0);
        else
            FragColor = vec4(color, clamp(exen_contour, 0.0, 1.0));
    }else{
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}  