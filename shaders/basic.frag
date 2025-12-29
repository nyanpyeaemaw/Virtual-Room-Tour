#version 330 core

in vec3 fragPos;
in vec3 normal;
in vec2 texCoord;

out vec4 FragColor;

uniform sampler2D tex0;
uniform int   useTexture;
uniform vec3  objectColor;
uniform int   emissive;
uniform vec3  emissionColor;

#define MAX_LIGHTS 16
uniform int   numLights;
uniform vec3  lightPos[MAX_LIGHTS];
uniform vec3  lightColor[MAX_LIGHTS];
uniform float lightIntensity[MAX_LIGHTS];

uniform vec3  viewPos;
uniform float ambientBase;
uniform float surfaceGain;

/* New but optional controls (safe defaults applied below via consts)
   Set from main if you like:
   uniform int   sRGBTextures;    // 1 = decode texture from sRGB
   uniform float exposure;        // scene exposure before tone map
   uniform float lightRange;      // soft radius of point lights
*/
uniform int   sRGBTextures;
uniform float exposure;
uniform float lightRange;

// -------- helpers --------
vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 toSRGB (vec3 c) { return pow(c, vec3(1.0/2.2)); }

// ACES-ish tone map (Narkowicz 2015)
vec3 ACES(vec3 x){
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x*(a*x + b))/(x*(c*x + d) + e), 0.0, 1.0);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0){
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    // Defaults if you don't set uniforms from code
    int   _sRGBTex   = (sRGBTextures == 0 && sRGBTextures == 1) ? sRGBTextures : 1;
    float _exposure  = (exposure > 0.0) ? exposure : 1.0;
    float _range     = (lightRange > 0.0) ? lightRange : 8.0;

    vec3 N = normalize(normal);
    vec3 V = normalize(viewPos - fragPos);

    // Base albedo
    vec3 albedo = (useTexture == 1) ? texture(tex0, texCoord).rgb : objectColor;
    if (_sRGBTex == 1) albedo = toLinear(albedo);

    // Energy-conserving base (Lambert diffuse)
    vec3  Lsum = albedo * ambientBase;

    // Assume dielectric (plastic/paint) F0 ~ 0.04; tint slightly by albedo
    vec3  F0 = mix(vec3(0.04), albedo, 0.02);

    for (int i = 0; i < numLights; ++i) {
        vec3  Lvec = lightPos[i] - fragPos;
        float dist = max(length(Lvec), 1e-4);
        vec3  L    = Lvec / dist;

        // Inverse-square falloff with a soft rolloff near range
        float invSq = 1.0 / (dist * dist);
        float roll  = smoothstep(_range, 0.0, dist); // 1 near light, 0 past range
        float atten = invSq * roll;

        // N·L
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL <= 1e-4) continue;

        // Fresnel for specular
        vec3  H     = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        vec3  F     = fresnelSchlick(VdotH, F0);

        // Energy split: kS = specular, kD = diffuse (non-metal)
        vec3  kS = F;
        vec3  kD = (vec3(1.0) - kS);

        // Simple Blinn-Phong lobe (you can swap for GGX later)
        float shininess = 64.0;             // tweak if you like
        float specPow   = pow(NdotH, shininess);
        vec3  specular  = specPow * lightColor[i] * 0.5; // 0.5: tame highlight

        vec3  diffuse   = (albedo / 3.14159265) * NdotL; // Lambert / PI

        // Per-light contribution
        vec3  Li = (kD * diffuse + specular) * lightIntensity[i] * lightColor[i] * atten;
        Lsum += Li;
    }

    // Emissive
    if (emissive == 1) {
        // Emissive is already in linear
        Lsum += emissionColor;
    } else {
        // Only dim NON-emissive surfaces
        Lsum *= surfaceGain;
    }

    // Tone map + gamma
    vec3 mapped = ACES(Lsum * _exposure);
    FragColor = vec4(toSRGB(mapped), 1.0);
}
