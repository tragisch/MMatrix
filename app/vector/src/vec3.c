/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "vec3.h"

#include <math.h>
#include <stddef.h>
#include <stddef.h>

static const float VEC3_EPSILON = 1e-8f;

Vec3 vec3_make(float x, float y, float z) {
  Vec3 vec = {x, y, z};
  return vec;
}

Vec3 vec3_add(Vec3 lhs, Vec3 rhs) {
  return vec3_make(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

Vec3 vec3_sub(Vec3 lhs, Vec3 rhs) {
  return vec3_make(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

Vec3 vec3_scale(Vec3 vec, float scalar) {
  return vec3_make(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

float vec3_dot(Vec3 lhs, Vec3 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 vec3_cross(Vec3 lhs, Vec3 rhs) {
  return vec3_make(lhs.y * rhs.z - lhs.z * rhs.y,
                   lhs.z * rhs.x - lhs.x * rhs.z,
                   lhs.x * rhs.y - lhs.y * rhs.x);
}

float vec3_length(Vec3 vec) { return sqrtf(vec3_dot(vec, vec)); }

float vec3_distance(Vec3 lhs, Vec3 rhs) {
  return vec3_length(vec3_sub(lhs, rhs));
}

bool vec3_normalize(Vec3 *vec) {
  if (vec == NULL) {
    return false;
  }
  float length = vec3_length(*vec);
  if (length <= VEC3_EPSILON) {
    return false;
  }
  const float inv = 1.0f / length;
  vec->x *= inv;
  vec->y *= inv;
  vec->z *= inv;
  return true;
}

Vec3 vec3_lerp(Vec3 lhs, Vec3 rhs, float t) {
  return vec3_add(lhs, vec3_scale(vec3_sub(rhs, lhs), t));
}

Vec3 vec3_project(Vec3 vec, Vec3 onto) {
  float denom = vec3_dot(onto, onto);
  if (denom <= VEC3_EPSILON) {
    return vec3_make(0.0f, 0.0f, 0.0f);
  }
  return vec3_scale(onto, vec3_dot(vec, onto) / denom);
}

Vec3 vec3_reflect(Vec3 incident, Vec3 normal) {
  Vec3 n = normal;
  if (!vec3_normalize(&n)) {
    return incident;
  }
  return vec3_sub(incident, vec3_scale(n, 2.0f * vec3_dot(incident, n)));
}
