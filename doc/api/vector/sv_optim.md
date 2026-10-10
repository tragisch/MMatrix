# `sv_optim.h` – Flat-Buffer Optimizers

The optimizer kernels update caller-owned, contiguous `float` buffers. Momentum
and second-moment buffers are also caller-owned and must be initialized by the
caller (normally to zero). All functions return `false` for invalid pointers,
lengths, or hyperparameters.

## SGD

`bool sv_optim_sgd(float *params, const float *grads, size_t n, float learning_rate, float grad_scale, float weight_decay)`

Updates `params -= learning_rate * (grads * grad_scale)`. Decoupled weight
decay is applied separately as `params -= learning_rate * weight_decay * params`;
pass zero to disable it.

## Momentum

`bool sv_optim_momentum(float *params, const float *grads, float *momentum_buffer, size_t n, float learning_rate, float momentum, float grad_scale, float weight_decay)`

Updates velocity as `momentum * velocity + grads * grad_scale`, then applies the
learning-rate step and optional decoupled weight decay.

## RMSprop

`bool sv_optim_rmsprop(float *params, const float *grads, float *squared_grad_buffer, size_t n, float learning_rate, float beta2, float epsilon, float grad_scale, float weight_decay)`

Updates the squared-gradient average and divides the scaled gradient by
`sqrt(average) + epsilon`.

## Adam

`bool sv_optim_adam(float *params, const float *grads, float *first_moment, float *second_moment, size_t n, float learning_rate, float beta1, float beta2, float epsilon, uint64_t step, float grad_scale, float weight_decay)`

Uses the nm_lib first/second moments and bias corrections based on the one-based
`step`. The denominator is `sqrt(v_hat) + epsilon`.

## Gradient norm clipping

`float sv_clip_norm(float *grads, size_t n, float max_norm)`

Returns the original L2 norm and scales gradients only when that norm exceeds
`max_norm`. An empty buffer returns zero; invalid arguments return `-1.0f`.
Norm accumulation uses a fixed scalar
order; scaling is elementwise.

The optimizer loops use OpenMP only when `n > 1,048,576`, avoiding parallel-region
overhead for small models.