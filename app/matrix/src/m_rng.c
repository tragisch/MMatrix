/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_rng.h"

uint64_t m_rng_resolve_seed(uint64_t requested_seed, uint64_t global_seed,
                            bool seed_initialized, uint64_t fallback_seed) {
  if (requested_seed != 0) {
    return requested_seed;
  }
  if (seed_initialized) {
    return global_seed;
  }
  return fallback_seed;
}
