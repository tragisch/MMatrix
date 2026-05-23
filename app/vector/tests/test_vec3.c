/*
 * SPDX-License-Identifier: MIT
 */

#include "vec3.h"

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_vec3_cross_and_dot_should_work(void) {
  Vec3 x = vec3_make(1.0f, 0.0f, 0.0f);
  Vec3 y = vec3_make(0.0f, 1.0f, 0.0f);
  Vec3 z = vec3_cross(x, y);

  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, vec3_dot(x, y));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, z.x);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, z.y);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, z.z);
}

void test_vec3_normalize_project_and_reflect_should_work(void) {
  Vec3 v = vec3_make(3.0f, 4.0f, 0.0f);
  Vec3 onto = vec3_make(1.0f, 0.0f, 0.0f);
  Vec3 incident = vec3_make(1.0f, -1.0f, 0.0f);
  Vec3 normal = vec3_make(0.0f, 1.0f, 0.0f);

  TEST_ASSERT_TRUE(vec3_normalize(&v));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, vec3_length(v));

  Vec3 projected = vec3_project(vec3_make(2.0f, 3.0f, 0.0f), onto);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 2.0f, projected.x);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.0f, projected.y);

  Vec3 reflected = vec3_reflect(incident, normal);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, reflected.x);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, reflected.y);
}
