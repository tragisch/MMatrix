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