# C API Design Guidelines (Production-Grade)

These guidelines define the required design rules for a professional, stable, and maintainable C library API.

They apply to all public headers and exported functions.

---

## 1. Naming Convention

All public functions MUST follow:

```
<namespace>_<action>
```
or in some cases. Ask 

```
<namespace>_<object>_<action>
```

Examples:

```
nm_init
nm_train
nm_predict
sm_multiply
```

DO NOT use:

```
init_network
train
predict
```

Types MUST use PascalCase with namespace prefix:

```
NmNetwork
NmNetworkConfig
```

Enums MUST use uppercase prefix:

```
NM_STATUS_OK
NM_ACT_RELU
```

---

## 2. Builder Pattern with Config Structs

All configurable objects MUST use configuration structs.

Provide defaults:

```c
NmNetworkConfig config = nm_network_config_defaults();
```

User modifies only required fields:

```c
config.learning_rate = 0.001f;
```

Initialize using config:

```c
nm_network_init(&net, &config);
```

DO NOT use wrapper APIs like `_easy`.

---

## 3. Lifecycle Symmetry

Every init MUST have destroy:

```
nm_network_init / nm_network_destroy
nm_object_create / nm_object_free
```

Destroy functions MUST be safe on partially initialized objects.

---

## 4. Error Handling Rule

High-level functions MUST return a status code.

Example:

```c
NmStatus nm_network_predict(
    const NmNetwork *net,
    const FloatMatrix *input,
    FloatMatrix **out_prediction
);
```

DO NOT return:

```
bool
NULL
```

Use status codes consistently.

---

## 5. Output via Pointer Parameters

Functions returning objects MUST use output pointers:

```c
NmStatus nm_object_create(Object **out_object);
```

Caller owns returned object unless documented otherwise.

---

## 6. Ownership Rule

Ownership MUST be explicit and documented.

Rules:

- Caller owns objects returned via output pointer
- Caller must destroy using corresponding destroy function
- Functions MUST NOT free caller-owned objects unless documented

---

## 7. Const Correctness

All input parameters MUST be const unless modified:

```c
NmStatus nm_network_predict(
    const NmNetwork *net,
    const FloatMatrix *input,
    FloatMatrix **out_prediction
);
```

---

## 8. High-Level vs Low-Level Separation

Separate public API into:

### High-Level Operations

Stable, user-facing:

```
nm_init
nm_train
nm_predict
nm_destroy
```

These MUST:

- return status code
- avoid exposing internal implementation

### Low-Level Technical Functions

Internal or advanced operations:

```
nm_matrix_multiply
nm_activation_relu
nm_loss_mse
```

These MAY return direct values if safe.

---

## 9. Allocation Transparency

Avoid hidden allocations in performance-critical functions.

Provide `_into` variants when appropriate:

```
nm_matrix_multiply_into(...)
```

Caller controls memory allocation.

---

## 10. Namespace Isolation

All public symbols MUST use namespace prefix:

```
nm_
Nm
NM_
```

Avoid global symbol pollution.

---

## 11. Header Organization

Public headers MUST be structured in logical sections:

```
Types
Config structs
Lifecycle functions
High-level operations
Low-level operations
Utility functions
```

---

## 12. Thread Safety

Library MUST be safe when different object instances are used independently.

Global mutable state MUST be avoided or documented.

---

## 13. API Stability

Public APIs MUST NOT break backward compatibility.

New features MUST use new functions or struct fields.


## 14. Three-Level Function Architecture (MANDATORY)

All operations that modify or produce objects MUST follow a three-level function architecture.

This ensures maximum performance, flexibility, and clarity.

### Level 1 — Core Function (PRIMARY IMPLEMENTATION)

Core functions are the canonical implementation.

Requirements:

- MUST be public
- MUST return status code (NmStatus / SmStatus)
- MUST NOT allocate memory
- MUST write results into caller-provided output object
- MUST be the single source of implementation logic
- MUST be used internally by all wrappers

Naming pattern:

```
<namespace>_<object>_<action>_to
```

Example:

```c
SmStatus sm_matrix_add_to(
    FloatMatrix *out,
    const FloatMatrix *a,
    const FloatMatrix *b
);
```

---

### Level 2 — Convenience Wrapper (ALLOCATING)

Convenience functions allocate and return new objects.

Requirements:

- MUST be public
- MUST allocate memory
- MUST call the core function internally
- MUST return pointer or NULL on failure
- MUST NOT duplicate core logic

Naming pattern:

```
<namespace>_<object>_<action>
```

Example:

```c
FloatMatrix *sm_matrix_add(
    const FloatMatrix *a,
    const FloatMatrix *b
);
```

Implementation pattern:

```c
FloatMatrix *sm_matrix_add(const FloatMatrix *a, const FloatMatrix *b)
{
    FloatMatrix *out = sm_matrix_create(a->rows, a->cols);

    if (!out)
        return NULL;

    if (sm_matrix_add_to(out, a, b) != SM_STATUS_OK) {
        sm_matrix_destroy(out);
        return NULL;
    }

    return out;
}
```

---

### Level 3 — Inplace Wrapper (MODIFY EXISTING OBJECT)

Inplace functions modify an existing object.

Requirements:

- MUST be public
- MUST return status code
- MUST NOT allocate memory
- MUST call core function internally
- MUST NOT duplicate logic

Naming pattern:

```
<namespace>_<object>_inplace_<action>
```

Example:

```c
SmStatus sm_matrix_inplace_add(
    FloatMatrix *inout,
    const FloatMatrix *other
);
```

Implementation pattern:

```c
SmStatus sm_matrix_inplace_add(
    FloatMatrix *inout,
    const FloatMatrix *other
)
{
    return sm_matrix_add_to(inout, inout, other);
}
```

---

## 15. Status Type Definition (MANDATORY)

Libraries MUST define a namespace-specific status enum.

Example:

```c
typedef enum SmStatus {

    SM_STATUS_OK = 0,

    SM_STATUS_ERR_NULL,
    SM_STATUS_ERR_INVALID_ARGUMENT,
    SM_STATUS_ERR_DIMENSION_MISMATCH,
    SM_STATUS_ERR_ALLOCATION_FAILED,
    SM_STATUS_ERR_INTERNAL

} SmStatus;
```

Requirements:

- MUST use namespace prefix
- MUST use explicit enum type
- MUST be used by all core and inplace functions
- MUST NOT use bool for public error reporting

---

## 16. Static Helper Functions Rule (MANDATORY)

Internal helper functions MUST be declared static.

Requirements:

- MUST NOT appear in public headers
- MUST be defined only in .c files
- MUST use namespace prefix and internal indicator

Naming pattern:

```
static <return_type>
<namespace>_<object>_internal_<action>(...)
```

Example:

```c
static SmStatus sm_matrix_internal_validate_add(
    const FloatMatrix *a,
    const FloatMatrix *b,
    const FloatMatrix *out
);
```

Rationale:

- prevents symbol collisions
- enforces encapsulation
- prevents accidental public API exposure

---

## 17. Implementation Hierarchy Rule (MANDATORY)

Implementation MUST follow this dependency hierarchy:

```
Convenience Wrapper
    ↓
Inplace Wrapper
    ↓
Core Function
    ↓
Static Helper Functions
```

Rules:

- Core functions MUST NOT call convenience wrappers
- Core functions MAY call static helpers
- Convenience and inplace wrappers MUST call core functions
- Logic MUST exist only in core functions

---

## 18. Allocation Transparency Rule (STRICT)

Memory allocation MUST occur ONLY in:

- explicit create/init functions
- convenience wrapper functions

Memory allocation MUST NOT occur in:

- core functions
- inplace functions

This guarantees predictable performance and prevents hidden allocation overhead.

---

These rules ensure:

- predictable performance
- safe ownership semantics
- stable and extensible API
- compatibility with HPC, ML, and real-time systems

---


## Summary

Required characteristics:

- Consistent naming
- Explicit lifecycle management
- Explicit ownership
- Status-based error handling
- Config-based initialization
- Clear high-level and low-level separation
- Namespace isolation
- Stable and extensible design

---