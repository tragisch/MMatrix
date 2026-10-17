#include "st_bf16_utils.h"
#include <string.h>

uint16_t st_f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(float));
    return (uint16_t)(u >> 16);
}

float st_bf16_to_f32(uint16_t x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

void st_f32_to_bf16_bulk(const float *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = st_f32_to_bf16(src[i]);
    }
}

void st_bf16_to_f32_bulk(const uint16_t *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = st_bf16_to_f32(src[i]);
    }
}
