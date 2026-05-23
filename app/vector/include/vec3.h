/**
 * @file vec3.h
 * @brief Public API for fixed-size 3D geometry vectors.
 */

#ifndef VEC3_H
#define VEC3_H

#include <stdbool.h>

typedef struct Vec3 {
  float x;
  float y;
  float z;
} Vec3;

Vec3 vec3_make(float x, float y, float z);
Vec3 vec3_add(Vec3 lhs, Vec3 rhs);
Vec3 vec3_sub(Vec3 lhs, Vec3 rhs);
Vec3 vec3_scale(Vec3 vec, float scalar);
float vec3_dot(Vec3 lhs, Vec3 rhs);
Vec3 vec3_cross(Vec3 lhs, Vec3 rhs);
float vec3_length(Vec3 vec);
float vec3_distance(Vec3 lhs, Vec3 rhs);
bool vec3_normalize(Vec3 *vec);
Vec3 vec3_lerp(Vec3 lhs, Vec3 rhs, float t);
Vec3 vec3_project(Vec3 vec, Vec3 onto);
Vec3 vec3_reflect(Vec3 incident, Vec3 normal);

#endif  // VEC3_H
