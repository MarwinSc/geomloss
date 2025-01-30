#version 460

out vec4 FragColor;
  
in vec2 TexCoords;

uniform sampler2D screenTexture;
uniform sampler2D depthTexture;

uniform bool render_contour;
uniform bool contour_overlay;
uniform float contour_amp;

uniform vec3 bg_color;

uniform float offset_v;
uniform float offset_h;
//const float offset_v = 1.0 / (720.0 / 1.0);  
//const float offset_h = 1.0 / (1280.0 / 1.0);

void main()
{
    if (render_contour)
    {
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

        //vec3 sampleTex[9];
        //for(int i = 0; i < 9; i++)
        //{
        //    sampleTex[i] = vec3(texture(screenTexture, TexCoords.st + offsets[i]));
        //}
        //vec3 col = vec3(0.0);
        //for(int i = 0; i < 9; i++)
        //    col += sampleTex[i] * kernel[i];
        
        float sampleTex[25];
        for(int i = 0; i < 25; i++)
        {
            sampleTex[i] = float(texture(depthTexture, TexCoords.st + (offsets_5[i])));
        }
        float col = float(0.0);
        for(int i = 0; i < 25; i++)
            col += sampleTex[i] * kernel5[i];

        col = clamp(col * contour_amp, 0.0, 1.0);

        if (contour_overlay){
            vec3 color = vec3(texture(screenTexture, TexCoords));
            FragColor = vec4(color - col, 1.0);
        }else{
            FragColor = vec4(bg_color - col, 1.0);
        }

    }else{
        FragColor = texture(screenTexture, TexCoords);
    }
}  