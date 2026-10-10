/* Tests for the graph-interpreter tensor kernels (st_interp.h). */

#include "st.h"
#include "st_interp.h"

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
  const size_t shape_a[3] = {32, 129, 64};  /* 264k elements, odd middle */
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

static bool near_layernorm(float actual, float expected) {
  /* Task tolerance against the naive scalar loop (1e-5 rel + 1e-6 abs). */
  return fabsf(actual - expected) <= 1.0e-6f + 1.0e-5f * fabsf(expected);
}

static bool near_layernorm_sum8(float actual, float expected) {
  /* Tight tolerance against nm_lib's sum8 algorithm; the remaining slack
   * covers fma contraction differences between the library and this test
   * binary. */
  return fabsf(actual - expected) <= 1.0e-6f + 1.0e-6f * fabsf(expected);
}

/* nm_lib's run_layernorm verbatim (sum8 mean, eight-lane variance), so the
 * kernel is checked against the reference semantics itself. */
static void ref_layernorm_sum8(const float *in, const float *gamma,
                               const float *beta, size_t group, size_t groups,
                               float eps, float *out) {
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
    if (gamma != NULL && beta != NULL) {
      for (size_t i = 0; i < group; ++i) {
        y[i] = y[i] * inverse * gamma[i] + beta[i];
      }
    } else {
      for (size_t i = 0; i < group; ++i) {
        float value = y[i] * inverse;
        if (gamma != NULL) {
          value *= gamma[i];
        }
        if (beta != NULL) {
          value += beta[i];
        }
        y[i] = value;
      }
    }
  }
}

/* Naive scalar reference: one group at a time, plain sums. */
static void ref_layernorm(const float *in, const float *gamma,
                          const float *beta, size_t group, size_t groups,
                          float eps, float *out) {
  for (size_t g = 0; g < groups; ++g) {
    const float *x = in + g * group;
    float *y = out + g * group;
    float mean = 0.0f;
    for (size_t i = 0; i < group; ++i) {
      mean += x[i];
    }
    mean /= (float)group;
    float variance = 0.0f;
    for (size_t i = 0; i < group; ++i) {
      y[i] = x[i] - mean;
      variance += y[i] * y[i];
    }
    variance /= (float)group;
    const float inverse = 1.0f / sqrtf(variance + eps);
    for (size_t i = 0; i < group; ++i) {
      float value = y[i] * inverse;
      if (gamma != NULL) {
        value *= gamma[i];
      }
      if (beta != NULL) {
        value += beta[i];
      }
      y[i] = value;
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
  float *expected = malloc(x->numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected);
  ref_layernorm(x->values, gamma ? gamma->values : NULL,
                beta ? beta->values : NULL, group, groups, 1.0e-5f, expected);

  /* nm_lib reference semantics (sum8), tight tolerance. */
  float *expected_sum8 = malloc(x->numel * sizeof(float));
  TEST_ASSERT_NOT_NULL(expected_sum8);
  ref_layernorm_sum8(x->values, gamma ? gamma->values : NULL,
                     beta ? beta->values : NULL, group, groups, 1.0e-5f,
                     expected_sum8);

  FloatTensor *out = st_clone(x);
  TEST_ASSERT_NOT_NULL(out);
  TEST_ASSERT_TRUE(st_layernorm(x, gamma, beta, norm_rank, 1.0e-5f, out));
  for (size_t i = 0; i < x->numel; ++i) {
    TEST_ASSERT_TRUE(near_layernorm_sum8(out->values[i],
                                        expected_sum8[i]));
  }
  /* Task tolerance against the naive loop; for groups <= 8 the sum8 order
   * is bit-identical to the naive order, so this also pins the small
   * inputs to the plain scalar reference. */
  if (group <= 8) {
    for (size_t i = 0; i < x->numel; ++i) {
      TEST_ASSERT_TRUE(near_layernorm(out->values[i], expected[i]));
    }
  }

  /* In-place normalization must produce the same values. */
  FloatTensor *inplace = st_clone(x);
  TEST_ASSERT_NOT_NULL(inplace);
  TEST_ASSERT_TRUE(st_layernorm(inplace, gamma, beta, norm_rank, 1.0e-5f, inplace));
  for (size_t i = 0; i < x->numel; ++i) {
    TEST_ASSERT_TRUE(near_layernorm_sum8(inplace->values[i],
                                        expected_sum8[i]));
  }

  free(expected_sum8);
  free(expected);
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
  ref_layernorm(x_promoted->values, gamma_promoted->values,
                beta_promoted->values, 7, 3, 1e-5f, expected);
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
  const size_t shape[4] = {2, 4, 129, 128};  /* 264k input, 2064 slots */
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
