#version 460

out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D Texture;

uniform float offset_h;
uniform float offset_v;

uniform float sigma;

//declare uniforms
uniform vec2 dir;

uniform int kernelSize; // Total number of samples (must be odd)

// Function to compute Gaussian weight
float gaussian(float x, float sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * 3.141592653589793) * sigma);
}

void main() {
	//this will be our RGBA sum
	vec4 sum = vec4(0.0);
	
	//our original texcoord for this fragment
	vec2 tc = TexCoords.st;
    
	//the direction of our blur
	//(1.0, 0.0) -> x-axis blur
	//(0.0, 1.0) -> y-axis blur
	float hstep = dir.x;
	float vstep = dir.y;

    // Output the final color
    FragColor = sum;

	float weightSum = 0.0; // Normalization factor

	int halfKernel = kernelSize / 2; // Number of samples on each side

    // Compute Gaussian weights dynamically
    for (int i = -halfKernel; i <= halfKernel; i++) {
        float x = float(i);
        
        float weight = gaussian(x, sigma);
        
        vec2 offset = vec2(x * offset_h * hstep, x * offset_v * vstep);
        sum += texture(Texture, tc + offset) * weight;
        
        weightSum += weight;
    }

    // Normalize the final color
    FragColor = sum / weightSum;
}