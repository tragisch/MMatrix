/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#ifndef M_STATUS_H
#define M_STATUS_H

typedef enum MStatus {
  MSTATUS_OK = 0,
  MSTATUS_INVALID_ARGUMENT = 1,
  MSTATUS_IO_ERROR = 2,
  MSTATUS_ALLOC_FAILED = 3,
  MSTATUS_FORMAT_ERROR = 4,
  MSTATUS_UNSUPPORTED_TYPE = 5,
  MSTATUS_INTERNAL_ERROR = 6,
} MStatus;

const char *m_status_to_string(MStatus status);

#endif  // M_STATUS_H
