/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef APP_MATRIX_SRC_M_RNG_H_
#define APP_MATRIX_SRC_M_RNG_H_

#include <stdbool.h>
#include <stdint.h>

/*
 * Keep as header inline to avoid call overhead in random hot loops.
 */
static inline uint64_t m_rng_mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t m_rng_resolve_seed(uint64_t requested_seed, uint64_t global_seed,
                            bool seed_initialized, uint64_t fallback_seed);

#endif  // APP_MATRIX_SRC_M_RNG_H_
