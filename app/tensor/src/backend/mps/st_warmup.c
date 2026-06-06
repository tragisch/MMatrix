/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_warmup.c — st_mps_warmup_shapes() implementation.
 *
 * Pre-populates MPSGraph caches for Conv2D, MaxPool2D, AvgPool2D, and
 * BatchNorm2D so that the first real inference call uses the cached graph
 * rather than paying the full compilation cost.
 *
 * Activated automatically when MMATRIX_ST_MPS_WARMUP=1 is set in the
 * environment.  Can also be called directly by user code.
 */

#include "st.h"
#include "st_backend.h"

/* Forward-declare the pool/BN warmup functions from st_mps.m.
 * They are defined in the ObjC translation unit but have a pure-C ABI. */
void st_mps_warmup_maxpool2d(size_t n, size_t c, size_t h, size_t w,
                              size_t kh, size_t kw,
                              size_t sh, size_t sw,
                              size_t ph, size_t pw,
                              size_t oh, size_t ow);
void st_mps_warmup_avgpool2d(size_t n, size_t c, size_t h, size_t w,
                              size_t kh, size_t kw,
                              size_t sh, size_t sw,
                              size_t ph, size_t pw,
                              size_t oh, size_t ow);
void st_mps_warmup_batchnorm2d(size_t n, size_t c, size_t h, size_t w);

/* ------------------------------------------------------------------ */

void st_mps_warmup_shapes(const StWarmupShape *shapes, size_t count) {
  if (!shapes || count == 0) return;

  /* Quick check: if MPS backend is not registered, nothing to warm up. */
  if (!st_backend_mps()) return;

  for (size_t i = 0; i < count; ++i) {
    const StWarmupShape *s = &shapes[i];
    if (s->n == 0 || s->c_in == 0 || s->h == 0 || s->w == 0) continue;

    /* Conv2D warmup (skip when c_out == 0) */
    if (s->c_out > 0 && s->kh > 0 && s->kw > 0) {
      const size_t dh = 1, dw = 1;
      st_backend_mps_warmup_conv2d(s->n, s->c_in, s->h, s->w,
                                    s->c_out, s->kh, s->kw,
                                    s->sh ? s->sh : 1,
                                    s->sw ? s->sw : 1,
                                    s->ph, s->pw, dh, dw);
    }

    /* Pool warmup — compute output dims with stride/pad, kernel from s->kh */
    if (s->kh > 0 && s->kw > 0) {
      const size_t sh    = s->sh ? s->sh : 1;
      const size_t sw    = s->sw ? s->sw : 1;
      const size_t out_h = (s->h + 2 * s->ph - s->kh) / sh + 1;
      const size_t out_w = (s->w + 2 * s->pw - s->kw) / sw + 1;

      st_mps_warmup_maxpool2d(s->n, s->c_in, s->h, s->w,
                               s->kh, s->kw, sh, sw, s->ph, s->pw,
                               out_h, out_w);
      st_mps_warmup_avgpool2d(s->n, s->c_in, s->h, s->w,
                               s->kh, s->kw, sh, sw, s->ph, s->pw,
                               out_h, out_w);
    }

    /* BatchNorm2D warmup */
    st_mps_warmup_batchnorm2d(s->n, s->c_in, s->h, s->w);
  }
}
