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
