#ifndef ST_BF16_UTILS_H
#define ST_BF16_UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint16_t st_f32_to_bf16(float x);
float st_bf16_to_f32(uint16_t x);
void st_f32_to_bf16_bulk(const float *src, uint16_t *dst, size_t n);
void st_bf16_to_f32_bulk(const uint16_t *src, float *dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif // ST_BF16_UTILS_H