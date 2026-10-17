/* Exercise public APIs using installed headers and archives only. */
#include <assert.h>
#include <stddef.h>
#include "sm.h"
#include "dms.h"
#include "sv_optim.h"
#ifdef __APPLE__
#include "st.h"
#include "st_convert.h"
#include "st_interp.h"
#include "st_f16.h"
#endif

int main(void) {
  FloatMatrix *a = sm_create(2, 2);
  assert(a);
  for (size_t i = 0; i < 4; ++i) a->values[i] = (float)(i + 1);
  FloatMatrix *product = sm_multiply(a, a);
  assert(product && product->values[0] == 7.0f && product->values[3] == 22.0f);
  DoubleSparseMatrix *sparse = dms_create_identity(2);
  assert(sparse && dms_get(sparse, 1, 1) == 1.0);
  float parameter = 1.0f, gradient = 2.0f;
  assert(sv_optim_sgd(&parameter, &gradient, 1, 0.25f, 1.0f, 0.0f));
  assert(parameter == 0.5f);
#ifdef __APPLE__
  FloatTensor *x = st_from_sm(a);
  assert(x);
  FloatTensor *sum = st_binary_broadcast(ST_BIN_ADD, x, x);
  assert(sum);
  FloatMatrix *result = sm_from_st(sum);
  assert(result && result->values[3] == 8.0f);
  assert(st_f32_to_f16(1.0f) == 0x3c00);
  sm_destroy(result);
  st_destroy(sum);
  st_destroy(x);
#endif
  dms_destroy(sparse);
  sm_destroy(product);
  sm_destroy(a);
  return 0;
}
