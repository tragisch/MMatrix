/* Tests for the graph-interpreter tensor kernels (st_interp.h). */

#include "st.h"
#include "st_interp.h"
#include "st_shape_ops.h"

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "st_bf16_utils.h"

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 6
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- Helpers ----------------------------------------------------------- */

static float apply_ref_op(StBinOp op, float a, float b) {
  switch (op) {
    case ST_BIN_ADD:
      return a + b;
    case ST_BIN_SUB:
      return a - b;
    case ST_BIN_MUL:
      return a * b;
    default:
      return a / b;
  }
}

static float sample_value(size_t seed) {
  /* Deterministic mix of magnitudes and signs; some values are zero so
   * division produces IEEE infinities. */
  if (seed % 11 == 0) {
    return 0.0f;
  }
  const float x = (float)((int)(seed % 97) - 48) * 0.137f;
  return sinf((float)seed * 1.7f) * (x == 0.0f ? 1.0f : x);
}

static FloatTensor *make_tensor(size_t ndim, const size_t *shape) {
  FloatTensor *t = st_create(ndim, shape);
  TEST_ASSERT_NOT_NULL(t);
  for (size_t i = 0; i < t->numel; ++i) {
    t->values[i] = sample_value(i);
  }
  return t;
}

/* Naive scalar reference: NumPy broadcast, one element at a time. */
static void ref_binary(StBinOp op, const FloatTensor *a, const FloatTensor *b,
                       float *out, size_t o_ndim, const size_t *o_dims) {
  size_t o_numel = 1;
  for (size_t i = 0; i < o_ndim; ++i) {
    o_numel *= o_dims[i];
  }
  size_t index[ST_MAX_DIMS] = {0};
  for (size_t flat = 0; flat < o_numel; ++flat) {
    size_t ia = 0, ib = 0;
    size_t stride_a = 1, stride_b = 1;
    for (size_t d = o_ndim; d-- > 0;) {
      const size_t da = d < o_ndim - a->ndim ? 1 : a->shape[d - (o_ndim - a->ndim)];
      const size_t db = d < o_ndim - b->ndim ? 1 : b->shape[d - (o_ndim - b->ndim)];
      ia += index[d] % da * stride_a;
      ib += index[d] % db * stride_b;
      stride_a *= da;
      stride_b *= db;
    }
    out[flat] = apply_ref_op(op, a->values[ia], b->values[ib]);
    for (size_t d = o_ndim; d-- > 0;) {
      if (++index[d] < o_dims[d]) {
        break;
      }
      index[d] = 0;
    }
  }
}

static void check_binary_case(StBinOp op, const FloatTensor *a,
                              const FloatTensor *b) {
  size_t o_ndim = 0;
  size_t o_dims[ST_MAX_DIMS];
  const size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  for (size_t i = 0; i < ndim; ++i) {
    const size_t da = i < ndim - a->ndim ? 1 : a->shape[i - (ndim - a->ndim)];
    const size_t db = i < ndim - b->ndim ? 1 : b->shape[i - (ndim - b->ndim)];
    TEST_ASSERT_TRUE(da == db || da == 1 || db == 1);
    o_dims[i] = da == 1 ? db : da;
  }
  o_ndim = ndim;
  size_t o_numel = 1;
  for (size_t i = 0; i < o_ndim; ++i) {
    o_numel *= o_dims[i];
  }

  float *expected = malloc(o_numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_binary(op, a, b, expected, o_ndim, o_dims);

  FloatTensor *out = st_binary_broadcast(op, a, b);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_EQUAL_UINT(o_ndim, out->ndim);
  for (size_t i = 0; i < o_ndim; ++i) {
    TEST_ASSERT_EQUAL_UINT(o_dims[i], out->shape[i]);
  }
  TEST_ASSERT_EQUAL_UINT(ST_DTYPE_F32, out->dtype);
  TEST_ASSERT_EQUAL_MEMORY(expected, out->values, o_numel * sizeof(float));

  /* The _into variant writes the same bits into a prepared output. */
  FloatTensor *into = st_create(o_ndim, o_dims);
  TEST_ASSERT_NOT_NULL(into);
  for (size_t i = 0; i < into->numel; ++i) {
    into->values[i] = -12345.0f;
  }
  TEST_ASSERT_TRUE(st_binary_broadcast_into(op, a, b, into));
  TEST_ASSERT_EQUAL_MEMORY(expected, into->values, o_numel * sizeof(float));

  free(expected);
  st_destroy(into);
  st_destroy(out);
}

/* ---- Shape coverage ---------------------------------------------------- */

void test_st_binary_broadcast_covers_shapes_and_ops(void) {
  static const struct {
    size_t ndim_a;
    size_t a[ST_MAX_DIMS];
    size_t ndim_b;
    size_t b[ST_MAX_DIMS];
  } cases[] = {
      {.ndim_a = 4, .a = {2, 3, 4, 5}, .ndim_b = 4, .b = {2, 3, 4, 5}},
      {.ndim_a = 1, .a = {1}, .ndim_b = 2, .b = {3, 4}},
      {.ndim_a = 2, .a = {3, 4}, .ndim_b = 1, .b = {1}},
      {.ndim_a = 3, .a = {2, 3, 7}, .ndim_b = 1, .b = {7}},
      {.ndim_a = 4, .a = {2, 3, 4, 5}, .ndim_b = 2, .b = {4, 5}},
      {.ndim_a = 4, .a = {2, 3, 4, 5}, .ndim_b = 1, .b = {5}},
      {.ndim_a = 3, .a = {2, 1, 4}, .ndim_b = 3, .b = {2, 3, 1}},
      {.ndim_a = 3, .a = {2, 1, 4}, .ndim_b = 3, .b = {1, 3, 1}},
      {.ndim_a = 3, .a = {1, 3, 1}, .ndim_b = 3, .b = {2, 1, 5}},
      {.ndim_a = 1, .a = {7}, .ndim_b = 2, .b = {2, 7}},
      {.ndim_a = 3, .a = {2, 1, 5}, .ndim_b = 3, .b = {2, 3, 1}},
      {.ndim_a = 4, .a = {2, 2, 4, 4}, .ndim_b = 3, .b = {1, 1, 4}},
      {.ndim_a = 4, .a = {1, 1, 1, 1}, .ndim_b = 2, .b = {2, 3}},
  };
  const StBinOp ops[] = {ST_BIN_ADD, ST_BIN_SUB, ST_BIN_MUL, ST_BIN_DIV};

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    FloatTensor *a = make_tensor(cases[c].ndim_a, cases[c].a);
    FloatTensor *b = make_tensor(cases[c].ndim_b, cases[c].b);
    for (size_t o = 0; o < sizeof(ops) / sizeof(ops[0]); ++o) {
      check_binary_case(ops[o], a, b);
    }
    st_destroy(a);
    st_destroy(b);
  }
}

void test_st_binary_broadcast_covers_sizes(void) {
  static const size_t sizes[] = {1, 7, 999, 1000, 100001};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    const size_t shape_a[1] = {sizes[s]};
    const size_t shape_b[1] = {sizes[s]};
    FloatTensor *a = make_tensor(1, shape_a);
    FloatTensor *b = make_tensor(1, shape_b);
    check_binary_case(ST_BIN_ADD, a, b);
    check_binary_case(ST_BIN_DIV, a, b);
    st_destroy(a);
    st_destroy(b);
  }

  /* Zero-sized shapes are rejected at creation and by the kernels. */
  TEST_ASSERT_NULL(st_create(1, (size_t[]){0}));
}

void test_st_binary_broadcast_scalar_sizes_with_broadcast(void) {
  /* Broadcast against a scalar with odd lengths (SIMD tails). */
  static const size_t sizes[] = {1, 7, 1000, 100003};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    const size_t shape_a[1] = {sizes[s]};
    const size_t shape_one[1] = {1};
    FloatTensor *a = make_tensor(1, shape_a);
    FloatTensor *b = make_tensor(1, shape_one);
    check_binary_case(ST_BIN_MUL, a, b);
    check_binary_case(ST_BIN_SUB, b, a);
    st_destroy(a);
    st_destroy(b);
  }
}

/* ---- Error cases ------------------------------------------------------- */

void test_st_binary_broadcast_rejects_bad_input(void) {
  const size_t shape[2] = {2, 3};
  const size_t other[2] = {3, 4};
  FloatTensor *a = make_tensor(2, shape);
  FloatTensor *b = make_tensor(2, other);

  /* Shapes that do not broadcast. */
  TEST_ASSERT_NULL(st_binary_broadcast(ST_BIN_ADD, a, b));
  TEST_ASSERT_NULL(st_binary_broadcast(ST_BIN_ADD, b, a));

  /* NULL inputs. */
  TEST_ASSERT_NULL(st_binary_broadcast(ST_BIN_ADD, NULL, a));
  TEST_ASSERT_NULL(st_binary_broadcast(ST_BIN_ADD, a, NULL));

  /* Invalid op. */
  TEST_ASSERT_NULL(st_binary_broadcast((StBinOp)77, a, a));

  /* _into: wrong output shape, NULL, invalid op; nothing is written. */
  FloatTensor *out = st_create(2, other);
  TEST_ASSERT_NOT_NULL(out);
  for (size_t i = 0; i < out->numel; ++i) {
    out->values[i] = 42.0f;
  }
  TEST_ASSERT_FALSE(st_binary_broadcast_into(ST_BIN_ADD, a, b, out));
  TEST_ASSERT_FALSE(st_binary_broadcast_into(ST_BIN_ADD, NULL, b, out));
  TEST_ASSERT_FALSE(st_binary_broadcast_into(ST_BIN_ADD, a, b, NULL));
  TEST_ASSERT_FALSE(st_binary_broadcast_into((StBinOp)77, a, a, out));
  for (size_t i = 0; i < out->numel; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(42.0f, out->values[i]);
  }

  /* Non-contiguous tensors are rejected. */
  FloatTensor *base = make_tensor(2, (size_t[]){2, 3});
  const size_t view_shape[2] = {2, 2};
  const ptrdiff_t view_strides[2] = {3, 1};
  FloatTensor *view = st_view(base, 2, view_shape, view_strides, 0);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_NULL(st_binary_broadcast(ST_BIN_ADD, view, view));
  TEST_ASSERT_FALSE(st_binary_broadcast_into(ST_BIN_ADD, view, view, out));

  st_destroy(view);
  st_destroy(base);
  st_destroy(out);
  st_destroy(b);
  st_destroy(a);
}

/* ---- Determinism and bf16 ---------------------------------------------- */

void test_st_binary_broadcast_thread_count_bit_identical(void) {
  /* Above the OpenMP threshold the row split must not change any bit. */
  const size_t shape_a[3] = {32, 1025, 64};  /* > 2M elements, odd middle */
  const size_t shape_b[3] = {32, 1, 64};
  FloatTensor *a = make_tensor(3, shape_a);
  FloatTensor *b = make_tensor(3, shape_b);

  FloatTensor *one = NULL;
  FloatTensor *eight = NULL;
  omp_set_num_threads(1);
  one = st_binary_broadcast(ST_BIN_ADD, a, b);
  TEST_ASSERT_NOT_NULL(one);
  omp_set_num_threads(8);
  eight = st_binary_broadcast(ST_BIN_ADD, a, b);
  TEST_ASSERT_NOT_NULL(eight);
  TEST_ASSERT_EQUAL_MEMORY(one->values, eight->values,
                           one->numel * sizeof(float));

  st_destroy(eight);
  st_destroy(one);
  st_destroy(b);
  st_destroy(a);
}

void test_st_binary_broadcast_promotes_bf16(void) {
  const size_t shape[2] = {3, 7};
  FloatTensor *a_f32 = make_tensor(2, shape);
  FloatTensor *b_f32 = make_tensor(2, shape);
  FloatTensor *a = st_to_bf16(a_f32);
  FloatTensor *b = st_to_bf16(b_f32);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  /* bf16 inputs promote to f32; the result equals the f32 computation on
   * the promoted (quantized) inputs. */
  FloatTensor *a_promoted = st_to_f32(a);
  FloatTensor *b_promoted = st_to_f32(b);
  TEST_ASSERT_NOT_NULL(a_promoted);
  TEST_ASSERT_NOT_NULL(b_promoted);
  float *expected = malloc(a_promoted->numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_binary(ST_BIN_ADD, a_promoted, b_promoted, expected, 2, shape);

  FloatTensor *out = st_binary_broadcast(ST_BIN_ADD, a, b);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_EQUAL_UINT(ST_DTYPE_F32, out->dtype);
  TEST_ASSERT_EQUAL_MEMORY(expected, out->values, out->numel * sizeof(float));

  /* bf16 output is rounded back with the st conversion helpers. */
  FloatTensor *out_bf16 = st_create_bf16(2, shape);
  TEST_ASSERT_NOT_NULL(out_bf16);
  TEST_ASSERT_TRUE(st_binary_broadcast_into(ST_BIN_ADD, a_promoted,
                                            b_promoted, out_bf16));
  uint16_t *expected_bf16 = malloc(out->numel * sizeof(uint16_t));
  TEST_ASSERT_NOT_NULL(expected_bf16);
  st_f32_to_bf16_bulk(expected, expected_bf16, out->numel);
  TEST_ASSERT_EQUAL_MEMORY(expected_bf16, out_bf16->values,
                           out->numel * sizeof(uint16_t));

  free(expected_bf16);
  free(expected);
  st_destroy(out_bf16);
  st_destroy(out);
  st_destroy(b_promoted);
  st_destroy(a_promoted);
  st_destroy(b);
  st_destroy(a);
  st_destroy(b_f32);
  st_destroy(a_f32);
}

/* ---- LayerNorm ---------------------------------------------------------- */

/* Tolerance relative to the reference INTERMEDIATE magnitude: the
 * library's -ffast-math build rounds the apply chain differently from this
 * -O0 reference (reassociation and FMA), each difference bounded by an ulp
 * of the operands. A result-relative tolerance blows up on outputs where
 * the scaled value cancels against beta, so the reference also reports
 * |t| + |b| (t = y * inv * scale) and the check scales with that. */
static bool near_layernorm_sum8(float actual, float expected, float bound) {
  return fabsf(actual - expected) <= 1.0e-6f + 1.0e-5f * bound;
}

/* nm_lib's run_layernorm verbatim (sum8 mean, eight-lane variance), so the
 * kernel is checked against the reference semantics itself. `bound`
 * (optional) receives |y * inv * scale| + |beta| per element for the
 * intermediate-magnitude tolerance. */
static void ref_layernorm_sum8(const float *in, const float *gamma,
                               const float *beta, size_t group, size_t groups,
                               float eps, float *out, float *bound) {
  for (size_t g = 0; g < groups; ++g) {
    const float *x = in + g * group;
    float *y = out + g * group;
    float lanes[8] = {0.0f};
    size_t j = 0;
    for (; j + 8 <= group; j += 8) {
      for (size_t l = 0; l < 8; ++l) {
        lanes[l] += x[j + l];
      }
    }
    float mean = 0.0f;
    for (; j < group; ++j) {
      mean += x[j];
    }
    for (size_t l = 0; l < 8; ++l) {
      mean += lanes[l];
    }
    mean /= (float)group;
    for (size_t i = 0; i < group; ++i) {
      y[i] = x[i] - mean;
    }
    float var_lanes[8] = {0.0f};
    j = 0;
    for (; j + 8 <= group; j += 8) {
      for (size_t l = 0; l < 8; ++l) {
        var_lanes[l] += y[j + l] * y[j + l];
      }
    }
    float variance = 0.0f;
    for (; j < group; ++j) {
      variance += y[j] * y[j];
    }
    for (size_t l = 0; l < 8; ++l) {
      variance += var_lanes[l];
    }
    variance /= (float)group;
    const float inverse = 1.0f / sqrtf(variance + eps);
    for (size_t i = 0; i < group; ++i) {
      float t = y[i] * inverse;
      if (gamma != NULL) {
        t *= gamma[i];
      }
      if (beta != NULL) {
        y[i] = t + beta[i];
      } else {
        y[i] = t;
      }
      if (bound != NULL) {
        bound[g * group + i] =
            fabsf(t) + (beta != NULL ? fabsf(beta[i]) : 0.0f);
      }
    }
  }
}

static void check_layernorm_case(const FloatTensor *x, const FloatTensor *gamma,
                                 const FloatTensor *beta, size_t norm_rank) {
  size_t group = 1;
  for (size_t i = x->ndim - norm_rank; i < x->ndim; ++i) {
    group *= x->shape[i];
  }
  const size_t groups = x->numel / group;

  /* nm_lib reference semantics (sum8), tolerance relative to the
   * intermediate magnitude (see near_layernorm_sum8). */
  float *expected_sum8 = malloc(x->numel * sizeof(float));
  float *bound = malloc(x->numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected_sum8);
  TEST_ASSERT_NOT_NULL(bound);
  ref_layernorm_sum8(x->values, gamma ? gamma->values : NULL,
                     beta ? beta->values : NULL, group, groups, 1.0e-5f,
                     expected_sum8, bound);

  FloatTensor *out = st_clone(x);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_TRUE(st_layernorm(x, gamma, beta, norm_rank, 1.0e-5f, out));
  for (size_t i = 0; i < x->numel; ++i) {
    TEST_ASSERT_TRUE(
        near_layernorm_sum8(out->values[i], expected_sum8[i], bound[i]));
  }
  /* In-place normalization must produce the same values. */
  FloatTensor *inplace = st_clone(x);
  TEST_ASSERT_NOT_NULL(inplace);
  TEST_ASSERT_TRUE(st_layernorm(inplace, gamma, beta, norm_rank, 1.0e-5f, inplace));
  for (size_t i = 0; i < x->numel; ++i) {
    TEST_ASSERT_TRUE(near_layernorm_sum8(inplace->values[i],
                                        expected_sum8[i], bound[i]));
  }

  free(bound);
  free(expected_sum8);
  st_destroy(inplace);
  st_destroy(out);
}

static size_t group_numel(const FloatTensor *x, size_t norm_rank) {
  size_t group = 1;
  for (size_t i = x->ndim - norm_rank; i < x->ndim; ++i) {
    group *= x->shape[i];
  }
  return group;
}

void test_st_layernorm_matches_reference(void) {
  struct {
    size_t ndim;
    size_t shape[ST_MAX_DIMS];
    size_t norm_rank;
  } cases[] = {
      {1, {7}, 1},
      {2, {3, 7}, 1},
      {2, {97, 103}, 1},
      {4, {2, 3, 4, 5}, 1},
      {4, {2, 3, 4, 5}, 2},
      {4, {2, 3, 4, 5}, 3},
      {3, {1, 16, 64}, 1},
      {3, {16, 65, 32}, 2},
      {1, {100001}, 1},
      {3, {4, 128, 256}, 1},
      {1, {1}, 1},
  };
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    FloatTensor *x = make_tensor(cases[c].ndim, cases[c].shape);
    const size_t group = group_numel(x, cases[c].norm_rank);
    FloatTensor *gamma_full = make_tensor(1, (size_t[1]){group});
    FloatTensor *beta_full = make_tensor(1, (size_t[1]){group});

    check_layernorm_case(x, NULL, NULL, cases[c].norm_rank);
    check_layernorm_case(x, gamma_full, NULL, cases[c].norm_rank);
    check_layernorm_case(x, NULL, beta_full, cases[c].norm_rank);
    check_layernorm_case(x, gamma_full, beta_full, cases[c].norm_rank);

    st_destroy(beta_full);
    st_destroy(gamma_full);
    st_destroy(x);
  }
}

void test_st_layernorm_rejects_bad_input(void) {
  const size_t shape[2] = {3, 7};
  FloatTensor *x = make_tensor(2, shape);
  FloatTensor *out = make_tensor(2, shape);
  const size_t group_shape[1] = {7};
  FloatTensor *gamma = make_tensor(1, group_shape);
  const size_t wrong[1] = {6};
  FloatTensor *wrong_gamma = make_tensor(1, wrong);
  const size_t other_shape[2] = {3, 6};
  FloatTensor *wrong_out = make_tensor(2, other_shape);
  FloatTensor *view_base = make_tensor(2, (size_t[]){2, 3});
  const size_t view_shape[2] = {2, 2};
  const ptrdiff_t view_strides[2] = {3, 1};
  FloatTensor *view = st_view(view_base, 2, view_shape, view_strides, 0);

  TEST_ASSERT_FALSE(st_layernorm(x, gamma, NULL, 0, 1e-5f, out));
  TEST_ASSERT_FALSE(st_layernorm(x, gamma, NULL, 3, 1e-5f, out));
  TEST_ASSERT_FALSE(st_layernorm(x, NULL, NULL, 1, -1.0f, out));
  TEST_ASSERT_FALSE(st_layernorm(x, NULL, NULL, 1, NAN, out));
  TEST_ASSERT_FALSE(st_layernorm(x, NULL, NULL, 1, 1e-5f, wrong_out));
  TEST_ASSERT_FALSE(st_layernorm(x, wrong_gamma, NULL, 1, 1e-5f, out));
  TEST_ASSERT_FALSE(st_layernorm(x, NULL, wrong_gamma, 1, 1e-5f, out));
  TEST_ASSERT_FALSE(st_layernorm(NULL, NULL, NULL, 1, 1e-5f, out));
  TEST_ASSERT_FALSE(st_layernorm(x, NULL, NULL, 1, 1e-5f, NULL));
  TEST_ASSERT_FALSE(st_layernorm(view, NULL, NULL, 1, 1e-5f, out));
  TEST_ASSERT_FALSE(st_layernorm(x, NULL, NULL, 1, 1e-5f, view));

  st_destroy(view);
  st_destroy(view_base);
  st_destroy(wrong_out);
  st_destroy(wrong_gamma);
  st_destroy(gamma);
  st_destroy(out);
  st_destroy(x);
}

void test_st_layernorm_thread_count_bit_identical(void) {
  /* Above the parallel threshold; the eight-partial-sum order must keep
   * results bit-identical across thread counts. */
  const size_t shape[3] = {32, 129, 64};  /* 264k elements */
  FloatTensor *x = make_tensor(3, shape);
  const size_t group_shape[1] = {64};
  FloatTensor *gamma = make_tensor(1, group_shape);
  FloatTensor *beta = make_tensor(1, group_shape);

  FloatTensor *one = st_clone(x);
  FloatTensor *eight = st_clone(x);
  omp_set_num_threads(1);
  TEST_ASSERT_TRUE(st_layernorm(x, gamma, beta, 1, 1e-5f, one));
  omp_set_num_threads(8);
  TEST_ASSERT_TRUE(st_layernorm(x, gamma, beta, 1, 1e-5f, eight));
  TEST_ASSERT_EQUAL_MEMORY(one->values, eight->values,
                            one->numel * sizeof(float));

  st_destroy(eight);
  st_destroy(one);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(x);
}

void test_st_layernorm_promotes_bf16(void) {
  const size_t shape[2] = {3, 7};
  FloatTensor *x_f32 = make_tensor(2, shape);
  const size_t group_shape[1] = {7};
  FloatTensor *gamma_f32 = make_tensor(1, group_shape);
  FloatTensor *beta_f32 = make_tensor(1, group_shape);
  FloatTensor *x = st_to_bf16(x_f32);
  FloatTensor *gamma = st_to_bf16(gamma_f32);
  FloatTensor *beta = st_to_bf16(beta_f32);
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(gamma);
  TEST_ASSERT_NOT_NULL(beta);

  /* Reference on the promoted (quantized) inputs, then round to bf16. */
  FloatTensor *x_promoted = st_to_f32(x);
  FloatTensor *gamma_promoted = st_to_f32(gamma);
  FloatTensor *beta_promoted = st_to_f32(beta);
  float *expected = malloc(x->numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_layernorm_sum8(x_promoted->values, gamma_promoted->values,
                    beta_promoted->values, 7, 3, 1e-5f, expected, NULL);
  uint16_t *expected_bf16 = malloc(x->numel * sizeof(uint16_t));
  TEST_ASSERT_NOT_NULL(expected_bf16);
  st_f32_to_bf16_bulk(expected, expected_bf16, x->numel);

  FloatTensor *out = st_create_bf16(2, shape);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_TRUE(st_layernorm(x, gamma, beta, 1, 1e-5f, out));
  TEST_ASSERT_EQUAL_MEMORY(expected_bf16, out->values,
                           x->numel * sizeof(uint16_t));

  free(expected_bf16);
  free(expected);
  st_destroy(out);
  st_destroy(beta_promoted);
  st_destroy(gamma_promoted);
  st_destroy(x_promoted);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(x);
  st_destroy(beta_f32);
  st_destroy(gamma_f32);
  st_destroy(x_f32);
}

/* ---- Reduce max ---------------------------------------------------------- */

static uint32_t float_bits_of(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/* Naive reference with the documented NaN behavior: the first NaN in scan
 * order wins, otherwise the finite maximum with -INFINITY start. */
static float ref_reduce_slot(const float *x, size_t n) {
  float best = -INFINITY;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t bits = float_bits_of(x[i]);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
      return x[i];
    }
    if (x[i] > best) {
      best = x[i];
    }
  }
  return best;
}

/* Flat walk over the input in memory order; the first NaN of a slot in
 * scan order sticks, otherwise the finite maximum with -INFINITY start.
 * This matches the kernel's documented scan order. */
static void ref_reduce_max(const FloatTensor *x, uint32_t mask, float *out,
                           size_t out_numel) {
  for (size_t i = 0; i < out_numel; ++i) {
    out[i] = -INFINITY;
  }
  size_t index[ST_MAX_DIMS] = {0};
  for (size_t flat = 0; flat < x->numel; ++flat) {
    size_t slot = 0;
    size_t slot_stride = 1;
    for (size_t a = x->ndim; a-- > 0;) {
      if ((mask >> a) & 1u) {
        continue;
      }
      slot += index[a] * slot_stride;
      slot_stride *= x->shape[a];
    }
    const uint32_t out_bits = float_bits_of(out[slot]);
    const bool out_is_nan = (out_bits & 0x7f800000u) == 0x7f800000u &&
                            (out_bits & 0x007fffffu) != 0;
    if (out_is_nan) {
      continue;  /* the first NaN already won */
    }
    const float value = x->values[flat];
    const uint32_t bits = float_bits_of(value);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
      out[slot] = value;  /* first NaN in scan order */
    } else if (value > out[slot]) {
      out[slot] = value;
    }
    for (size_t a = x->ndim; a-- > 0;) {
      if (++index[a] < x->shape[a]) {
        break;
      }
      index[a] = 0;
    }
  }
}

static void check_reduce_case(const FloatTensor *x, uint32_t mask,
                              bool keepdims) {
  FloatTensor *actual = st_reduce_max(x, mask, keepdims);
  TEST_ASSERT_NOT_NULL(actual);
  size_t expected_numel = 1;
  for (size_t a = 0; a < x->ndim; ++a) {
    if ((mask >> a) & 1u) {
      if (keepdims) {
        /* still a 1-axis */
      }
    } else {
      expected_numel *= x->shape[a];
    }
  }
  TEST_ASSERT_EQUAL_UINT(expected_numel, actual->numel);
  float *expected = malloc(expected_numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_reduce_max(x, mask, expected, expected_numel);
  TEST_ASSERT_EQUAL_MEMORY(expected, actual->values,
                           expected_numel * sizeof(float));

  /* _into with keepdims shape must produce the same bits. */
  size_t into_dims[ST_MAX_DIMS];
  size_t into_ndim = 0;
  for (size_t a = 0; a < x->ndim; ++a) {
    if ((mask >> a) & 1u) {
      if (keepdims) {
        into_dims[into_ndim++] = 1;
      }
    } else {
      into_dims[into_ndim++] = x->shape[a];
    }
  }
  FloatTensor *into = st_create(into_ndim == 0 ? 1 : into_ndim,
                                into_ndim == 0 ? (size_t[1]){1} : into_dims);
  TEST_ASSERT_NOT_NULL(into);
  TEST_ASSERT_TRUE(st_reduce_max_into(x, mask, keepdims, into));
  TEST_ASSERT_EQUAL_MEMORY(expected, into->values,
                           expected_numel * sizeof(float));

  free(expected);
  st_destroy(into);
  st_destroy(actual);
}

void test_st_reduce_max_matches_reference(void) {
  struct {
    size_t ndim;
    size_t shape[ST_MAX_DIMS];
    uint32_t mask;
  } cases[] = {
      {1, {7}, 0x1},
      {1, {1000}, 0x1},
      {1, {100001}, 0x1},
      {2, {2, 3}, 0x1},
      {2, {2, 3}, 0x2},
      {2, {2, 3}, 0x3},
      {2, {97, 103}, 0x2},
      {3, {2, 3, 4}, 0x1},
      {3, {2, 3, 4}, 0x2},
      {3, {2, 3, 4}, 0x4},
      {3, {2, 3, 4}, 0x5},
      {3, {2, 3, 4}, 0x7},
      {4, {2, 3, 4, 5}, 0x8},
      {4, {2, 3, 4, 5}, 0x3},
      {4, {2, 3, 4, 5}, 0xf},
      {4, {2, 4, 129, 128}, 0x8},
      {3, {4, 128, 256}, 0x4},
      {3, {4, 128, 256}, 0x2},
  };
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    FloatTensor *x = make_tensor(cases[c].ndim, cases[c].shape);
    check_reduce_case(x, cases[c].mask, false);
    check_reduce_case(x, cases[c].mask, true);
    st_destroy(x);
  }
}

void test_st_reduce_max_empty_mask_copies_input(void) {
  const size_t shape[2] = {3, 5};
  FloatTensor *x = make_tensor(2, shape);
  FloatTensor *out = st_reduce_max(x, 0, false);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_EQUAL_UINT(2, out->ndim);
  TEST_ASSERT_EQUAL_MEMORY(x->values, out->values, x->numel * sizeof(float));
  st_destroy(out);
  st_destroy(x);
}

void test_st_reduce_max_propagates_nan(void) {
  /* Documented behavior: a NaN in a reduction group makes the output slot
   * the bit pattern of the first NaN in scan order; other slots stay
   * finite. */
  const size_t shape[2] = {3, 4};
  FloatTensor *x = make_tensor(2, shape);
  const uint32_t nan_bits_a = 0x7fc00001u;
  const uint32_t nan_bits_b = 0x7f800100u;
  float nan_a, nan_b;
  memcpy(&nan_a, &nan_bits_a, sizeof(nan_a));
  memcpy(&nan_b, &nan_bits_b, sizeof(nan_b));
  x->values[1 * 4 + 2] = nan_a;  /* slot 1 */
  x->values[2 * 4 + 0] = nan_b;  /* slot 2 */
  x->values[2 * 4 + 3] = nan_a;  /* slot 2, later NaN: must not win */

  FloatTensor *out = st_reduce_max(x, 0x2, false);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_EQUAL_UINT(3, out->numel);
  TEST_ASSERT_EQUAL_UINT32(nan_bits_a, float_bits_of(out->values[1]));
  TEST_ASSERT_EQUAL_UINT32(nan_bits_b, float_bits_of(out->values[2]));
  TEST_ASSERT_EQUAL_FLOAT(3.0f, out->values[0]);

  st_destroy(out);
  st_destroy(x);
}

void test_st_reduce_max_keeps_infinity(void) {
  const size_t shape[2] = {2, 3};
  FloatTensor *x = make_tensor(2, shape);
  for (size_t i = 0; i < x->numel; ++i) {
    x->values[i] = -INFINITY;
  }
  x->values[4] = INFINITY;
  FloatTensor *out = st_reduce_max(x, 0x2, false);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_EQUAL_FLOAT(-INFINITY, out->values[0]);
  TEST_ASSERT_EQUAL_FLOAT(INFINITY, out->values[1]);
  st_destroy(out);
  st_destroy(x);
}

void test_st_reduce_max_rejects_bad_input(void) {
  const size_t shape[2] = {2, 3};
  FloatTensor *x = make_tensor(2, shape);
  const size_t wrong_shape[2] = {3, 2};
  FloatTensor *wrong = make_tensor(2, wrong_shape);

  /* Mask bit at or above the rank. */
  TEST_ASSERT_NULL(st_reduce_max(x, 0x4, false));
  TEST_ASSERT_NULL(st_reduce_max(x, 0x8, true));
  TEST_ASSERT_NULL(st_reduce_max(NULL, 0x1, false));
  TEST_ASSERT_FALSE(st_reduce_max_into(x, 0x4, false, wrong));
  TEST_ASSERT_FALSE(st_reduce_max_into(x, 0x1, false, NULL));
  TEST_ASSERT_FALSE(st_reduce_max_into(NULL, 0x1, false, wrong));

  /* Non-contiguous input is rejected. */
  FloatTensor *base = make_tensor(2, (size_t[]){2, 3});
  const size_t view_shape[2] = {2, 2};
  const ptrdiff_t view_strides[2] = {3, 1};
  FloatTensor *view = st_view(base, 2, view_shape, view_strides, 0);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_NULL(st_reduce_max(view, 0x1, false));
  st_destroy(view);
  st_destroy(base);

  st_destroy(wrong);
  st_destroy(x);
}

void test_st_reduce_max_thread_count_bit_identical(void) {
  /* Above the parallel threshold; slots are independent so results must be
   * bit-identical across thread counts. */
  const size_t shape[4] = {2, 4, 1025, 128};  /* > 1M input, 8200 slots */
  FloatTensor *x = make_tensor(4, shape);
  omp_set_num_threads(1);
  FloatTensor *one = st_reduce_max(x, 0x8, false);
  omp_set_num_threads(8);
  FloatTensor *eight = st_reduce_max(x, 0x8, false);
  TEST_ASSERT_NOT_NULL(one);
  TEST_ASSERT_NOT_NULL(eight);
  TEST_ASSERT_EQUAL_MEMORY(one->values, eight->values,
                           one->numel * sizeof(float));
  st_destroy(eight);
  st_destroy(one);
  st_destroy(x);
}

void test_st_reduce_max_promotes_bf16(void) {
  const size_t shape[2] = {3, 7};
  FloatTensor *x_f32 = make_tensor(2, shape);
  FloatTensor *x = st_to_bf16(x_f32);
  TEST_ASSERT_NOT_NULL(x);
  FloatTensor *promoted = st_to_f32(x);
  TEST_ASSERT_NOT_NULL(promoted);

  float *expected = malloc(3 * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  for (size_t slot = 0; slot < 3; ++slot) {
    expected[slot] = ref_reduce_slot(promoted->values + slot * 7, 7);
  }
  FloatTensor *out = st_reduce_max(x, 0x2, false);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_EQUAL_UINT(ST_DTYPE_F32, out->dtype);
  TEST_ASSERT_EQUAL_MEMORY(expected, out->values, 3 * sizeof(float));

  FloatTensor *out_bf16 = st_create_bf16(1, (size_t[1]){3});
  TEST_ASSERT_NOT_NULL(out_bf16);
  TEST_ASSERT_TRUE(st_reduce_max_into(x, 0x2, false, out_bf16));
  FloatTensor *back = st_to_f32(out_bf16);
  TEST_ASSERT_NOT_NULL(back);
  for (size_t slot = 0; slot < 3; ++slot) {
    const uint16_t expected_bf16 = st_f32_to_bf16(expected[slot]);
    const uint16_t actual_bf16 = ((const uint16_t *)out_bf16->values)[slot];
    TEST_ASSERT_EQUAL_UINT16(expected_bf16, actual_bf16);
  }

  free(expected);
  st_destroy(back);
  st_destroy(out_bf16);
  st_destroy(out);
  st_destroy(promoted);
  st_destroy(x);
  st_destroy(x_f32);
}

/* ---- Batched matmul ------------------------------------------------------ */

/* Naive batched matmul reference. */
static void ref_matmul(const FloatTensor *a, const FloatTensor *b,
                       FloatTensor *c) {
  const size_t m = a->shape[a->ndim - 2];
  const size_t k = a->shape[a->ndim - 1];
  const size_t n = b->shape[b->ndim - 1];
  const size_t batch_a = a->ndim - 2;
  const size_t batch_b = b->ndim - 2;
  const size_t batch_rank = batch_a > batch_b ? batch_a : batch_b;
  size_t d[ST_MAX_DIMS];
  size_t da[ST_MAX_DIMS];
  size_t db[ST_MAX_DIMS];
  for (size_t i = 0; i < batch_rank; ++i) {
    da[i] = i < batch_rank - batch_a ? 1 : a->shape[i - (batch_rank - batch_a)];
    db[i] = i < batch_rank - batch_b ? 1 : b->shape[i - (batch_rank - batch_b)];
    d[i] = da[i] == 1 ? db[i] : da[i];
  }
  size_t batches = 1;
  for (size_t i = 0; i < batch_rank; ++i) {
    batches *= d[i];
  }
  const bool single_b = b->numel == k * n;
  for (size_t index = 0; index < batches; ++index) {
    size_t rest = index;
    size_t offset_a = 0;
    size_t offset_b = 0;
    size_t scale_a = 1;
    size_t scale_b = 1;
    for (size_t i = batch_rank; i-- > 0;) {
      const size_t coordinate = rest % d[i];
      rest /= d[i];
      offset_a += (da[i] == 1 ? 0 : coordinate) * scale_a;
      offset_b += (db[i] == 1 ? 0 : coordinate) * scale_b;
      scale_a *= da[i];
      scale_b *= db[i];
    }
    const float *pa = a->values + offset_a * m * k;
    const float *pb = b->values + (single_b ? 0 : offset_b * k * n);
    float *pc = c->values + index * m * n;
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t p = 0; p < k; ++p) {
          sum += pa[i * k + p] * pb[p * n + j];
        }
        pc[i * n + j] = sum;
      }
    }
  }
}

static bool near_matmul(float actual, float expected) {
  /* BLAS accumulation order differs from the naive reference. */
  return fabsf(actual - expected) <= 1.0e-4f * (1.0f + fabsf(expected));
}

static void check_matmul_case(const FloatTensor *a, const FloatTensor *b) {
  const size_t m = a->shape[a->ndim - 2];
  const size_t k = a->shape[a->ndim - 1];
  const size_t n = b->shape[b->ndim - 1];
  const size_t batch_a = a->ndim - 2;
  const size_t batch_b = b->ndim - 2;
  const size_t batch_rank = batch_a > batch_b ? batch_a : batch_b;
  size_t out_dims[ST_MAX_DIMS];
  size_t out_ndim = 0;
  for (size_t i = 0; i < batch_rank; ++i) {
    const size_t da =
        i < batch_rank - batch_a ? 1 : a->shape[i - (batch_rank - batch_a)];
    const size_t db =
        i < batch_rank - batch_b ? 1 : b->shape[i - (batch_rank - batch_b)];
    out_dims[out_ndim++] = da == 1 ? db : da;
  }
  out_dims[out_ndim++] = m;
  out_dims[out_ndim++] = n;

  FloatTensor *c = st_create(out_ndim, out_dims);
  TEST_ASSERT_NOT_NULL(c);
  FloatTensor *expected = st_create(out_ndim, out_dims);
  TEST_ASSERT_NOT_NULL(expected);
  ref_matmul(a, b, expected);

  TEST_ASSERT_TRUE(st_matmul_batched(a, b, c));
  for (size_t i = 0; i < c->numel; ++i) {
    TEST_ASSERT_TRUE(near_matmul(c->values[i], expected->values[i]));
  }

  st_destroy(expected);
  st_destroy(c);
}

void test_st_matmul_batched_matches_reference(void) {
  static const struct {
    size_t ndim_a;
    size_t a[ST_MAX_DIMS];
    size_t ndim_b;
    size_t b[ST_MAX_DIMS];
  } cases[] = {
      {.ndim_a = 2, .a = {7, 5}, .ndim_b = 2, .b = {5, 3}},
      {.ndim_a = 2, .a = {1, 1}, .ndim_b = 2, .b = {1, 1}},
      {.ndim_a = 2, .a = {16, 65}, .ndim_b = 2, .b = {65, 32}},
      {.ndim_a = 3, .a = {3, 4, 5}, .ndim_b = 2, .b = {5, 7}},
      {.ndim_a = 3, .a = {2, 3, 7}, .ndim_b = 3, .b = {2, 7, 5}},
      {.ndim_a = 3, .a = {2, 3, 7}, .ndim_b = 2, .b = {7, 5}},
      {.ndim_a = 4, .a = {2, 3, 4, 5}, .ndim_b = 4, .b = {2, 3, 5, 7}},
      {.ndim_a = 4, .a = {2, 3, 4, 5}, .ndim_b = 4, .b = {2, 1, 5, 7}},
      {.ndim_a = 4, .a = {2, 1, 4, 5}, .ndim_b = 4, .b = {2, 3, 5, 7}},
      {.ndim_a = 4, .a = {2, 1, 4, 5}, .ndim_b = 3, .b = {2, 5, 7}},
      {.ndim_a = 3, .a = {1, 16, 64}, .ndim_b = 2, .b = {64, 64}},
      {.ndim_a = 3, .a = {4, 128, 256}, .ndim_b = 2, .b = {256, 64}},
      {.ndim_a = 4, .a = {2, 4, 128, 64}, .ndim_b = 4, .b = {2, 4, 64, 128}},
      {.ndim_a = 4, .a = {1, 4, 16, 64}, .ndim_b = 4, .b = {1, 4, 64, 16}},
      {.ndim_a = 2, .a = {97, 3}, .ndim_b = 2, .b = {3, 101}},
  };
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    FloatTensor *a = make_tensor(cases[c].ndim_a, cases[c].a);
    FloatTensor *b = make_tensor(cases[c].ndim_b, cases[c].b);
    check_matmul_case(a, b);
    st_destroy(b);
    st_destroy(a);
  }
}

void test_st_matmul_batched_rejects_bad_input(void) {
  FloatTensor *a = make_tensor(2, (size_t[2]){3, 5});
  FloatTensor *bad_k = make_tensor(2, (size_t[2]){4, 7});
  FloatTensor *rank1 = make_tensor(1, (size_t[1]){5});
  FloatTensor *c = make_tensor(2, (size_t[2]){3, 7});
  FloatTensor *wrong_c = make_tensor(2, (size_t[2]){7, 3});
  const size_t b_shape[2] = {5, 7};

  TEST_ASSERT_FALSE(st_matmul_batched(a, bad_k, c));
  TEST_ASSERT_FALSE(st_matmul_batched(rank1, rank1, c));
  TEST_ASSERT_FALSE(st_matmul_batched(a, rank1, c));
  TEST_ASSERT_FALSE(st_matmul_batched(a, a, wrong_c));
  TEST_ASSERT_FALSE(st_matmul_batched(NULL, a, c));
  TEST_ASSERT_FALSE(st_matmul_batched(a, NULL, c));
  TEST_ASSERT_FALSE(st_matmul_batched(a, a, NULL));

  /* Nothing written on failure. */
  for (size_t i = 0; i < wrong_c->numel; ++i) {
    wrong_c->values[i] = 42.0f;
  }
  TEST_ASSERT_FALSE(st_matmul_batched(a, bad_k, c) == false && false);
  TEST_ASSERT_FALSE(st_matmul_batched(a, bad_k, wrong_c));
  for (size_t i = 0; i < wrong_c->numel; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(42.0f, wrong_c->values[i]);
  }

  /* Non-contiguous operands are rejected. */
  FloatTensor *base = make_tensor(2, (size_t[2]){2, 3});
  const size_t view_shape[2] = {2, 2};
  const ptrdiff_t view_strides[2] = {3, 1};
  FloatTensor *view = st_view(base, 2, view_shape, view_strides, 0);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_FALSE(st_matmul_batched(view, view, c));
  st_destroy(view);
  st_destroy(base);

  st_destroy(wrong_c);
  st_destroy(c);
  st_destroy(bad_k);
  st_destroy(a);
  (void)b_shape;
}

void test_st_matmul_batched_thread_count_within_tolerance(void) {
  /* MatMul is the one interpreter kernel that is NOT bit-identical across
   * thread counts: the BLAS backend's accumulation order may differ, so
   * the comparison uses the documented 1e-4 relative tolerance instead. */
  /* Eight batches, each below the 262144-work parallel GEMM ceiling. */
  const size_t a_shape[4] = {2, 4, 63, 32};
  const size_t b_shape[4] = {2, 4, 32, 129};
  FloatTensor *a = make_tensor(4, a_shape);
  FloatTensor *b = make_tensor(4, b_shape);
  const size_t c_shape[4] = {2, 4, 63, 129};
  FloatTensor *one = st_create(4, c_shape);
  FloatTensor *eight = st_create(4, c_shape);
  TEST_ASSERT_NOT_NULL(one);
  TEST_ASSERT_NOT_NULL(eight);

  omp_set_num_threads(1);
  TEST_ASSERT_TRUE(st_matmul_batched(a, b, one));
  omp_set_num_threads(8);
  TEST_ASSERT_TRUE(st_matmul_batched(a, b, eight));
  for (size_t i = 0; i < one->numel; ++i) {
    TEST_ASSERT_TRUE(near_matmul(one->values[i], eight->values[i]));
  }

  st_destroy(eight);
  st_destroy(one);
  st_destroy(b);
  st_destroy(a);
}

void test_st_matmul_batched_promotes_bf16(void) {
  const size_t a_shape[3] = {2, 3, 5};
  const size_t b_shape[2] = {5, 7};
  FloatTensor *a_f32 = make_tensor(3, a_shape);
  FloatTensor *b_f32 = make_tensor(2, b_shape);
  FloatTensor *a = st_to_bf16(a_f32);
  FloatTensor *b = st_to_bf16(b_f32);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);
  FloatTensor *a_promoted = st_to_f32(a);
  FloatTensor *b_promoted = st_to_f32(b);
  TEST_ASSERT_NOT_NULL(a_promoted);
  TEST_ASSERT_NOT_NULL(b_promoted);

  /* Reference on the promoted inputs. */
  const size_t c_shape[3] = {2, 3, 7};
  FloatTensor *expected = st_create(3, c_shape);
  TEST_ASSERT_NOT_NULL(expected);
  ref_matmul(a_promoted, b_promoted, expected);

  /* f32 output from bf16 operands. */
  FloatTensor *c = st_create(3, c_shape);
  TEST_ASSERT_NOT_NULL(c);
  TEST_ASSERT_TRUE(st_matmul_batched(a, b, c));
  for (size_t i = 0; i < c->numel; ++i) {
    TEST_ASSERT_TRUE(near_matmul(c->values[i], expected->values[i]));
  }

  /* bf16 output is rounded back. */
  FloatTensor *c_bf16 = st_create_bf16(3, c_shape);
  TEST_ASSERT_NOT_NULL(c_bf16);
  TEST_ASSERT_TRUE(st_matmul_batched(a_promoted, b_promoted, c_bf16));
  for (size_t i = 0; i < c->numel; ++i) {
    const uint16_t expected_bf16 = st_f32_to_bf16(expected->values[i]);
    const uint16_t actual_bf16 = ((const uint16_t *)c_bf16->values)[i];
    /* bf16 rounds to 8 mantissa bits; equal bits or equal promoted values
     * within the matmul tolerance both pass. */
    if (expected_bf16 != actual_bf16) {
      const float e = st_bf16_to_f32(expected_bf16);
      const float g = st_bf16_to_f32(actual_bf16);
      TEST_ASSERT_TRUE(fabsf(e - g) <= 1.0e-2f * (1.0f + fabsf(e)));
    }
  }

  st_destroy(c_bf16);
  st_destroy(c);
  st_destroy(expected);
  st_destroy(b_promoted);
  st_destroy(a_promoted);
  st_destroy(b);
  st_destroy(a);
  st_destroy(b_f32);
  st_destroy(a_f32);
}

/* ---- Gather -------------------------------------------------------------- */

/* Naive gather reference: out[o, j, inner] = table[o, indices[j], inner]. */
static void ref_gather(const FloatTensor *table, const int64_t *indices,
                       size_t count, size_t axis, float *out) {
  size_t outer = 1;
  size_t inner = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer *= table->shape[i];
  }
  for (size_t i = axis + 1; i < table->ndim; ++i) {
    inner *= table->shape[i];
  }
  const size_t dim = table->shape[axis];
  for (size_t o = 0; o < outer; ++o) {
    for (size_t j = 0; j < count; ++j) {
      const int64_t index = indices[j] < 0 ? indices[j] + (int64_t)dim : indices[j];
      memcpy(out + (o * count + j) * inner,
             table->values + (o * dim + (size_t)index) * inner,
             inner * sizeof(float));
    }
  }
}

static void check_gather_case(const FloatTensor *table, const int64_t *indices,
                              size_t count, size_t axis) {
  size_t out_dims[ST_MAX_DIMS];
  memcpy(out_dims, table->shape, table->ndim * sizeof(size_t));
  out_dims[axis] = count;
  FloatTensor *out = st_create(table->ndim, out_dims);
  TEST_ASSERT_NOT_NULL(out);
  float *expected = malloc(st_tensor_numel(out) * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_gather(table, indices, count, axis, expected);

  TEST_ASSERT_TRUE(st_gather(table, indices, count, axis, out));
  TEST_ASSERT_EQUAL_MEMORY(expected, out->values,
                           st_tensor_numel(out) * sizeof(float));
  free(expected);
  st_destroy(out);
}

void test_st_gather_matches_reference(void) {
  const size_t table_shapes[][4] = {
      {7}, {16, 7}, {100, 7, 3}, {2, 3, 5, 7}, {30000, 16},
  };
  const size_t ranks[] = {1, 2, 3, 4, 2};
  for (size_t c = 0; c < sizeof(ranks) / sizeof(ranks[0]); ++c) {
    FloatTensor *table = make_tensor(ranks[c], table_shapes[c]);
    for (size_t axis = 0; axis < ranks[c]; ++axis) {
      const int64_t dim = (int64_t)table_shapes[c][axis];
      const int64_t half = dim / 2;
      const int64_t indices[] = {0, 1, -1, half, -half - 1, dim - 1, -dim};
      const size_t count = sizeof(indices) / sizeof(indices[0]);
      check_gather_case(table, indices, count, axis);
    }
    st_destroy(table);
  }

  /* Larger count on a realistic embedding table. */
  FloatTensor *table = make_tensor(2, (size_t[2]){30000, 64});
  int64_t indices[512];
  for (size_t i = 0; i < 512; ++i) {
    indices[i] = (int64_t)((i * 7919) % 30000);
    if (i % 3 == 0) {
      indices[i] -= 30000;  /* negative form */
    }
  }
  check_gather_case(table, indices, 512, 0);
  st_destroy(table);
}

void test_st_gather_rejects_bad_input(void) {
  FloatTensor *table = make_tensor(2, (size_t[2]){4, 3});
  const size_t out_shape[2] = {2, 3};
  FloatTensor *out = make_tensor(2, out_shape);
  const int64_t ok_indices[2] = {0, 1};
  const int64_t bad_high[2] = {0, 4};
  const int64_t bad_low[2] = {0, -5};

  /* Out-of-range indices fail without writing. */
  for (size_t i = 0; i < out->numel; ++i) {
    out->values[i] = 42.0f;
  }
  TEST_ASSERT_FALSE(st_gather(table, bad_high, 2, 0, out));
  TEST_ASSERT_FALSE(st_gather(table, bad_low, 2, 0, out));
  for (size_t i = 0; i < out->numel; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(42.0f, out->values[i]);
  }

  /* count == 0, NULL, wrong axis, wrong shape, non-contiguous. */
  TEST_ASSERT_FALSE(st_gather(table, ok_indices, 0, 0, out));
  TEST_ASSERT_FALSE(st_gather(NULL, ok_indices, 2, 0, out));
  TEST_ASSERT_FALSE(st_gather(table, NULL, 2, 0, out));
  TEST_ASSERT_FALSE(st_gather(table, ok_indices, 2, 2, out));
  TEST_ASSERT_FALSE(st_gather(table, ok_indices, 2, 1, out));
  FloatTensor *view_base = make_tensor(2, (size_t[2]){4, 4});
  const size_t view_shape[2] = {4, 2};
  const ptrdiff_t view_strides[2] = {4, 1};
  FloatTensor *view = st_view(view_base, 2, view_shape, view_strides, 0);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_FALSE(st_gather(view, ok_indices, 2, 0, out));
  st_destroy(view);
  st_destroy(view_base);

  /* Valid call still works after the failures. */
  TEST_ASSERT_TRUE(st_gather(table, ok_indices, 2, 0, out));

  st_destroy(out);
  st_destroy(table);
}

void test_st_gather_bytes_handles_i64_and_bool(void) {
  /* int64 table, gather along axis 0. */
  const size_t shape[2] = {5, 3};
  int64_t table[15];
  for (size_t i = 0; i < 15; ++i) {
    table[i] = (int64_t)(i * i) - 20;
  }
  const int64_t indices[4] = {4, -5, 2, 0};
  int64_t out[12];
  TEST_ASSERT_TRUE(st_gather_bytes(table, 2, shape, indices, 4, 0,
                                 sizeof(int64_t), out));
  for (size_t j = 0; j < 4; ++j) {
    const int64_t row = indices[j] < 0 ? indices[j] + 5 : indices[j];
    for (size_t e = 0; e < 3; ++e) {
      TEST_ASSERT_EQUAL_INT64(table[row * 3 + (int64_t)e], out[j * 3 + e]);
    }
  }

  /* bool table, gather along axis 1. */
  const size_t bool_shape[2] = {2, 5};
  uint8_t bools[10];
  for (size_t i = 0; i < 10; ++i) {
    bools[i] = (uint8_t)(i % 2);
  }
  const int64_t bool_indices[3] = {1, -1, 3};
  uint8_t bool_out[6];
  TEST_ASSERT_TRUE(st_gather_bytes(bools, 2, bool_shape, bool_indices, 3, 1, 1,
                                 bool_out));
  const size_t picked[3] = {1, 4, 3};
  for (size_t o = 0; o < 2; ++o) {
    for (size_t j = 0; j < 3; ++j) {
      TEST_ASSERT_EQUAL_UINT8(bools[o * 5 + picked[j]], bool_out[o * 3 + j]);
    }
  }

  TEST_ASSERT_FALSE(st_gather_bytes(table, 2, shape, indices, 4, 2, 8, out));
  TEST_ASSERT_FALSE(st_gather_bytes(table, 2, shape, indices, 0, 0, 8, out));
  TEST_ASSERT_FALSE(st_gather_bytes(NULL, 2, shape, indices, 4, 0, 8, out));
}

void test_st_gather_copies_bf16_bits(void) {
  const size_t shape[2] = {8, 5};
  FloatTensor *table_f32 = make_tensor(2, shape);
  FloatTensor *table = st_to_bf16(table_f32);
  TEST_ASSERT_NOT_NULL(table);
  const size_t out_shape[2] = {3, 5};
  FloatTensor *out = st_create_bf16(2, out_shape);
  TEST_ASSERT_NOT_NULL(out);
  const int64_t indices[3] = {7, 0, -4};
  TEST_ASSERT_TRUE(st_gather(table, indices, 3, 0, out));
  const uint16_t *table_bf16 = (const uint16_t *)table->values;
  const uint16_t *out_bf16 = (const uint16_t *)out->values;
  const size_t picked[3] = {7, 0, 4};
  for (size_t j = 0; j < 3; ++j) {
    for (size_t e = 0; e < 5; ++e) {
      TEST_ASSERT_EQUAL_UINT16(table_bf16[picked[j] * 5 + e],
                               out_bf16[j * 5 + e]);
    }
  }
  st_destroy(out);
  st_destroy(table);
  st_destroy(table_f32);
}

/* ---- Materializing permute ------------------------------------------------ */

static void ref_permute_copy(const FloatTensor *src, const size_t *perm,
                             float *out, size_t out_numel) {
  size_t index[ST_MAX_DIMS] = {0};
  for (size_t flat = 0; flat < out_numel; ++flat) {
    size_t src_index[ST_MAX_DIMS];
    for (size_t d = 0; d < src->ndim; ++d) {
      src_index[perm[d]] = index[d];
    }
    size_t src_flat = 0;
    for (size_t d = 0; d < src->ndim; ++d) {
      src_flat = src_flat * src->shape[d] + src_index[d];
    }
    out[flat] = src->values[src_flat];
    for (size_t d = src->ndim; d-- > 0;) {
      if (++index[d] < src->shape[perm[d]]) {
        break;
      }
      index[d] = 0;
    }
  }
}

static void check_permute_case(const FloatTensor *src, const size_t *perm) {
  size_t out_dims[ST_MAX_DIMS];
  for (size_t i = 0; i < src->ndim; ++i) {
    out_dims[i] = src->shape[perm[i]];
  }
  FloatTensor *out = st_create(src->ndim, out_dims);
  TEST_ASSERT_NOT_NULL(out);
  float *expected = malloc(src->numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_permute_copy(src, perm, expected, src->numel);

  TEST_ASSERT_TRUE(st_permute_copy_into(src, perm, out));
  TEST_ASSERT_EQUAL_MEMORY(expected, out->values, src->numel * sizeof(float));

  /* The view variant agrees on values. */
  FloatTensor *view = st_permute(src, perm);
  TEST_ASSERT_NOT_NULL(view);
  FloatTensor *materialized = st_to_f32(view);
  TEST_ASSERT_NOT_NULL(materialized);
  /* st_permute is a view; st_to_f32 materializes it. */
  TEST_ASSERT_EQUAL_MEMORY(expected, materialized->values,
                           src->numel * sizeof(float));

  free(expected);
  st_destroy(materialized);
  st_destroy(view);
  st_destroy(out);
}

void test_st_permute_copy_into_matches_reference(void) {
  const size_t shape4[4] = {2, 3, 4, 5};
  FloatTensor *x4 = make_tensor(4, shape4);
  check_permute_case(x4, (size_t[4]){0, 2, 1, 3});
  check_permute_case(x4, (size_t[4]){0, 2, 3, 1});
  check_permute_case(x4, (size_t[4]){3, 2, 1, 0});
  check_permute_case(x4, (size_t[4]){1, 0, 3, 2});
  st_destroy(x4);

  FloatTensor *x2 = make_tensor(2, (size_t[2]){97, 65});
  check_permute_case(x2, (size_t[2]){1, 0});
  st_destroy(x2);

  FloatTensor *x1 = make_tensor(1, (size_t[1]){7});
  check_permute_case(x1, (size_t[1]){0});
  st_destroy(x1);

  /* Attention-style [B, H, T, dk] -> [B, T, H, dk]. */
  FloatTensor *attn = make_tensor(4, (size_t[4]){2, 4, 128, 64});
  check_permute_case(attn, (size_t[4]){0, 2, 1, 3});
  st_destroy(attn);
}

void test_st_permute_copy_into_rejects_bad_input(void) {
  FloatTensor *src = make_tensor(3, (size_t[3]){2, 3, 4});
  FloatTensor *out = make_tensor(3, (size_t[3]){3, 2, 4});
  FloatTensor *wrong = make_tensor(3, (size_t[3]){2, 3, 4});

  TEST_ASSERT_FALSE(st_permute_copy_into(src, (size_t[3]){0, 1, 3}, out));
  TEST_ASSERT_FALSE(st_permute_copy_into(src, (size_t[3]){0, 1, 1}, out));
  TEST_ASSERT_FALSE(st_permute_copy_into(src, NULL, out));
  TEST_ASSERT_FALSE(st_permute_copy_into(src, (size_t[3]){1, 0, 2}, wrong));
  TEST_ASSERT_FALSE(st_permute_copy_into(NULL, (size_t[3]){1, 0, 2}, out));

  TEST_ASSERT_TRUE(st_permute_copy_into(src, (size_t[3]){1, 0, 2}, out));

  /* Non-contiguous tensors are rejected (views must be materialized
   * first, e.g. via st_to_f32 or the copy kernels themselves). */
  FloatTensor *view_base = make_tensor(2, (size_t[2]){2, 3});
  const size_t view_shape[2] = {2, 2};
  const ptrdiff_t view_strides[2] = {3, 1};
  FloatTensor *view = st_view(view_base, 2, view_shape, view_strides, 0);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_FALSE(st_permute_copy_into(view, (size_t[2]){1, 0}, out));
  TEST_ASSERT_FALSE(st_permute_copy_into(src, (size_t[3]){1, 0, 2}, view));
  st_destroy(view);
  st_destroy(view_base);

  st_destroy(wrong);
  st_destroy(out);
  st_destroy(src);
}

void test_st_permute_copy_into_bytes_handles_i64(void) {
  const size_t shape[3] = {2, 3, 4};
  int64_t src[24];
  for (size_t i = 0; i < 24; ++i) {
    src[i] = (int64_t)i;
  }
  int64_t out[24];
  const size_t perm[3] = {1, 2, 0};
  TEST_ASSERT_TRUE(
      st_permute_copy_into_bytes(src, 3, shape, perm, sizeof(int64_t), out));
  /* out[i1][i2][i0] = src[i0][i1][i2] */
  for (size_t i1 = 0; i1 < 3; ++i1) {
    for (size_t i2 = 0; i2 < 4; ++i2) {
      for (size_t i0 = 0; i0 < 2; ++i0) {
        TEST_ASSERT_EQUAL_INT64(src[i0 * 12 + i1 * 4 + i2],
                                out[i1 * 8 + i2 * 2 + i0]);
      }
    }
  }
  TEST_ASSERT_FALSE(st_permute_copy_into_bytes(src, 3, shape,
                                               (size_t[3]){0, 0, 1}, 8, out));
}

void test_st_copy_bytes_rejects_overflow_without_writing(void) {
  const uint8_t src[4] = {1, 2, 3, 4};
  uint8_t out[4] = {9, 9, 9, 9};
  const uint8_t expected[4] = {9, 9, 9, 9};
  const int64_t index = 0;
  const size_t gather_shape[3] = {2, SIZE_MAX / 2 + 1, 2};
  TEST_ASSERT_FALSE(st_gather_bytes(src, 3, gather_shape, &index, 1, 0, 1, out));
  TEST_ASSERT_FALSE(st_gather_bytes(src, 2, (size_t[2]){2, 2}, &index,
                                    SIZE_MAX / 2 + 1, 1, 1, out));
  TEST_ASSERT_FALSE(st_gather_bytes(src, 1, (size_t[1]){2}, &index, 1, 0,
                                    SIZE_MAX, out));
  const size_t shape[2] = {SIZE_MAX / 2 + 1, 2};
  const size_t perm[2] = {0, 1};
  TEST_ASSERT_FALSE(st_permute_copy_into_bytes(src, 2, shape, perm, 1, out));
  TEST_ASSERT_FALSE(st_permute_copy_into_bytes(
      src, 1, (size_t[1]){2}, (size_t[1]){0}, SIZE_MAX / 2 + 1, out));
  TEST_ASSERT_EQUAL_MEMORY(expected, out, sizeof(out));
}

void test_st_permute_copy_into_rejects_out_of_range_axis(void) {
  FloatTensor *src = make_tensor(2, (size_t[2]){2, 3});
  FloatTensor *out = make_tensor(2, (size_t[2]){3, 2});
  float before[6];
  memcpy(before, out->values, sizeof(before));
  TEST_ASSERT_FALSE(st_permute_copy_into(src, (size_t[2]){SIZE_MAX, 0}, out));
  TEST_ASSERT_FALSE(st_permute_copy_into(src, (size_t[2]){0, 0}, out));
  TEST_ASSERT_EQUAL_MEMORY(before, out->values, sizeof(before));
  st_destroy(out);
  st_destroy(src);
}

void test_st_permute_copy_bytes_preserves_unaligned_payloads(void) {
  const uint32_t bits[6] = {0x7fc12345u, 0x80000000u, 0xffffffffu,
                           0x12345678u, 0u, 0x7f800001u};
  uint8_t src[sizeof(bits) + 1];
  uint8_t out[sizeof(bits) + 1];
  memcpy(src + 1, bits, sizeof(bits));
  TEST_ASSERT_TRUE(st_permute_copy_into_bytes(
      src + 1, 2, (size_t[2]){2, 3}, (size_t[2]){1, 0}, 4, out + 1));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      TEST_ASSERT_EQUAL_MEMORY(bits + j * 3 + i, out + 1 + (i * 2 + j) * 4, 4);
    }
  }
  const uint64_t wide[6] = {0, UINT64_MAX, UINT64_C(0x7ff0000000000001),
                            UINT64_C(0x8000000000000000), 123, 456};
  uint8_t src_wide[sizeof(wide) + 1];
  uint8_t out_wide[sizeof(wide) + 1];
  memcpy(src_wide + 1, wide, sizeof(wide));
  TEST_ASSERT_TRUE(st_permute_copy_into_bytes(
      src_wide + 1, 2, (size_t[2]){2, 3}, (size_t[2]){1, 0}, 8, out_wide + 1));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      TEST_ASSERT_EQUAL_MEMORY(wide + j * 3 + i,
                               out_wide + 1 + (i * 2 + j) * 8, 8);
    }
  }
}

/* ---- Aliasing policy (in-place) ------------------------------------------ */

void test_st_binary_broadcast_into_allows_elementwise_inplace(void) {
  /* Documented aliasing rule: out may alias an operand whose shape equals
   * out's shape; the row kernels read and write the same element-wise
   * position. Partial overlap with a differently shaped operand stays
   * forbidden. */
  const size_t shape[2] = {3, 7};
  FloatTensor *a = make_tensor(2, shape);
  FloatTensor *b = make_tensor(2, shape);
  float *expected = malloc(30 * sizeof(float));  /* largest case below */
  TEST_ASSERT_NOT_NULL(expected);
  for (size_t i = 0; i < a->numel; ++i) {
    expected[i] = a->values[i] + b->values[i];
  }
  TEST_ASSERT_TRUE(st_binary_broadcast_into(ST_BIN_ADD, a, b, a));
  TEST_ASSERT_EQUAL_MEMORY(expected, a->values, a->numel * sizeof(float));

  /* Suffix bias path, in-place accumulation like `x += bias`. */
  const size_t x_shape[3] = {2, 3, 5};
  const size_t bias_shape[1] = {5};
  FloatTensor *x = make_tensor(3, x_shape);
  FloatTensor *bias = make_tensor(1, bias_shape);
  for (size_t i = 0; i < x->numel; ++i) {
    expected[i] = x->values[i] + bias->values[i % 5];
  }
  TEST_ASSERT_TRUE(st_binary_broadcast_into(ST_BIN_ADD, x, bias, x));
  TEST_ASSERT_EQUAL_MEMORY(expected, x->values, x->numel * sizeof(float));

  /* General walk path with a mid-broadcast operand, in-place. */
  const size_t mid_shape[3] = {1, 3, 1};
  FloatTensor *mid = make_tensor(3, mid_shape);
  for (size_t i = 0; i < x->numel; ++i) {
    expected[i] = x->values[i] * mid->values[(i / 5) % 3];
  }
  TEST_ASSERT_TRUE(st_binary_broadcast_into(ST_BIN_MUL, x, mid, x));
  TEST_ASSERT_EQUAL_MEMORY(expected, x->values, x->numel * sizeof(float));

  /* out aliases the full-size right operand. */
  FloatTensor *c = make_tensor(2, shape);
  FloatTensor *d = make_tensor(2, shape);
  for (size_t i = 0; i < c->numel; ++i) {
    expected[i] = c->values[i] - d->values[i];
  }
  TEST_ASSERT_TRUE(st_binary_broadcast_into(ST_BIN_SUB, c, d, d));
  TEST_ASSERT_EQUAL_MEMORY(expected, d->values, d->numel * sizeof(float));

  free(expected);
  st_destroy(d);
  st_destroy(c);
  st_destroy(mid);
  st_destroy(bias);
  st_destroy(x);
  st_destroy(b);
  st_destroy(a);
}
