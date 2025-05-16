/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm.h"
#include "sm.h"

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Creation of matrices:
 *******************************/

#define EPSILON 1e-5f

void setUp(void) {
  // Remove the test file if it exists
  // remove("test_matrix.mat");
}

// Teardown function called after each test
void tearDown(void) {
  // Remove the test file if it exists
  // remove("test_matrix.mat");
}

void test_nm_apply_relu_should_set_negatives_to_zero(void) {
  float values[2][2] = {{-2.0f, 0.0f}, {3.0f, -1.5f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_relu(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_nm_apply_sigmoid_should_map_to_range_0_to_1(void) {
  float values[2][2] = {{-2.0f, 0.0f}, {2.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_sigmoid(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f / (1.0f + expf(2.0f)),
                           sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.5f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f / (1.0f + expf(-2.0f)),
                           sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f / (1.0f + expf(-4.0f)),
                           sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_nm_apply_tanh_should_map_to_range_minus1_to_1(void) {
  float values[2][2] = {{-2.0f, 0.0f}, {2.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_tanh(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, tanhf(-2.0f), sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, tanhf(2.0f), sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, tanhf(4.0f), sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_nm_apply_softmax_should_yield_probabilities_that_sum_to_1(void) {
  float values[1][4] = {{1.0f, 2.0f, 3.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(1, 4, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_softmax(mat);

  float sum = 0.0f;
  for (size_t i = 0; i < 4; ++i) {
    float val = sm_get(mat, 0, i);
    TEST_ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
    sum += val;
  }
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sum);

  sm_destroy(mat);
}

void test_nm_inplace_add_rowwise_should_add_row_to_all_rows(void) {
  float values[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float bias[1][3] = {{10.0f, 20.0f, 30.0f}};

  FloatMatrix *mat = sm_from_array_static(2, 3, values);
  FloatMatrix *bias_row = sm_from_array_static(1, 3, bias);

  nm_inplace_add_rowwise(mat, bias_row);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 11.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 22.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 33.0f, sm_get(mat, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 14.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 25.0f, sm_get(mat, 1, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 36.0f, sm_get(mat, 1, 2));

  sm_destroy(mat);
  sm_destroy(bias_row);
}

void test_nm_linear_should_perform_multiplication_and_add_bias(void) {
  float input_vals[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  float weight_vals[2][2] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
  float bias_vals[1][2] = {{9.0f, 10.0f}};

  FloatMatrix *input = sm_from_array_static(2, 2, input_vals);
  FloatMatrix *weights = sm_from_array_static(2, 2, weight_vals);
  FloatMatrix *bias = sm_from_array_static(1, 2, bias_vals);

  FloatMatrix *result = nm_linear(input, weights, bias);
  TEST_ASSERT_NOT_NULL(result);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5 * 1 + 7 * 2 + 9,
                           sm_get(result, 0, 0)); // 5+14+9 = 28
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6 * 1 + 8 * 2 + 10,
                           sm_get(result, 0, 1)); // 6+16+10 = 32
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5 * 3 + 7 * 4 + 9,
                           sm_get(result, 1, 0)); // 15+28+9 = 52
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6 * 3 + 8 * 4 + 10,
                           sm_get(result, 1, 1)); // 18+32+10 = 60

  sm_destroy(input);
  sm_destroy(weights);
  sm_destroy(bias);
  sm_destroy(result);
}

void test_nm_mse_loss_should_return_mean_squared_error(void) {
  float pred_vals[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  float target_vals[2][2] = {{1.0f, 2.0f}, {2.0f, 4.0f}};
  FloatMatrix *pred = sm_from_array_static(2, 2, pred_vals);
  FloatMatrix *target = sm_from_array_static(2, 2, target_vals);

  float mse = nm_mse_loss(pred, target);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.25f, mse); // (0^2 + 0^2 + 1^2 + 0^2)/4
}

void test_nm_cross_entropy_loss_should_return_correct_value(void) {
  float pred_vals[1][3] = {{0.7f, 0.2f, 0.1f}};
  float target_vals[1][3] = {{1.0f, 0.0f, 0.0f}};
  FloatMatrix *pred = sm_from_array_static(1, 3, pred_vals);
  FloatMatrix *target = sm_from_array_static(1, 3, target_vals);

  float ce = nm_cross_entropy_loss(pred, target);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, -logf(0.7f), ce);
}

void test_nm_softmax_denominator_should_compute_correct_sum(void) {
  float vec[3] = {1.0f, 2.0f, 3.0f};
  float expected = expf(1.0f - 3.0f) + expf(2.0f - 3.0f) + expf(3.0f - 3.0f);
  float result = nm_softmax_denominator(vec, 3);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected, result);
}

void test_nm_active_library_should_return_non_null(void) {
  const char *lib = nm_active_library();
  printf("Active library: %s\n", lib);
  TEST_ASSERT_NOT_NULL(lib);
}

void test_nm_d_relu_should_zero_out_gradients_where_activation_is_nonpositive(
    void) {
  float act_vals[1][5] = {{-1.0f, 0.0f, 0.5f, 2.0f, -0.3f}};
  float grad_vals[1][5] = {{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}};

  FloatMatrix *act = sm_from_array_static(1, 5, act_vals);
  FloatMatrix *grad = sm_from_array_static(1, 5, grad_vals);

  nm_d_relu(act, grad);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(grad, 0, 0)); // a <= 0
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(grad, 0, 1)); // a == 0
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.3f, sm_get(grad, 0, 2)); // a > 0
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.4f, sm_get(grad, 0, 3)); // a > 0
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(grad, 0, 4)); // a < 0

  sm_destroy(act);
  sm_destroy(grad);
}

void test_nm_d_sigmoid_should_apply_chain_rule(void) {
  float act_vals[1][4] = {{0.0f, 0.5f, 1.0f, 0.8f}};
  float grad_vals[1][4] = {{1.0f, 1.0f, 1.0f, 2.0f}};

  FloatMatrix *act = sm_from_array_static(1, 4, act_vals);
  FloatMatrix *grad = sm_from_array_static(1, 4, grad_vals);

  nm_d_sigmoid(act, grad);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(grad, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.25f, sm_get(grad, 0, 1)); // 0.5 * 0.5
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(grad, 0, 2));  // 1 * 0
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.8f * 0.2f * 2.0f, sm_get(grad, 0, 3));

  sm_destroy(act);
  sm_destroy(grad);
}

void test_nm_d_tanh_should_apply_chain_rule(void) {
  float act_vals[1][4] = {{-1.0f, 0.0f, 0.5f, 0.8f}};
  float grad_vals[1][4] = {{1.0f, 1.0f, 2.0f, 2.0f}};

  FloatMatrix *act = sm_from_array_static(1, 4, act_vals);
  FloatMatrix *grad = sm_from_array_static(1, 4, grad_vals);

  nm_d_tanh(act, grad);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f - 1.0f, sm_get(grad, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f * (1.0f - 0.0f), sm_get(grad, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f * (1.0f - 0.25f), sm_get(grad, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f * (1.0f - 0.64f), sm_get(grad, 0, 3));

  sm_destroy(act);
  sm_destroy(grad);
}

void test_nm_d_softmax_crossentropy_should_compute_difference(void) {
  float pred_vals[1][3] = {{0.7f, 0.2f, 0.1f}};
  float target_vals[1][3] = {{1.0f, 0.0f, 0.0f}};
  float expected[1][3] = {{-0.3f, 0.2f, 0.1f}};

  FloatMatrix *pred = sm_from_array_static(1, 3, pred_vals);
  FloatMatrix *target = sm_from_array_static(1, 3, target_vals);
  FloatMatrix *grad = sm_create_zeros(1, 3);

  nm_d_softmax_crossentropy(pred, target, grad);

  for (size_t j = 0; j < 3; ++j) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[0][j], sm_get(grad, 0, j));
  }

  sm_destroy(pred);
  sm_destroy(target);
  sm_destroy(grad);
}

void test_dense_forward_should_compute_linear_activation_output(void) {
  float in_vals[1][2] = {{1.0f, 2.0f}};
  float w_vals[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
  float b_vals[1][2] = {{0.0f, 0.0f}};

  FloatMatrix *input = sm_from_array_static(1, 2, in_vals);
  FloatMatrix *weights = sm_from_array_static(2, 2, w_vals);
  FloatMatrix *bias = sm_from_array_static(1, 2, b_vals);

  DenseLayer layer = {.weights = weights,
                      .bias = bias,
                      .activation = nm_apply_relu,
                      .activation_derivative = nm_d_relu};

  FloatMatrix *output = dense_forward(&layer, input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_EQUAL_INT(1, output->rows);
  TEST_ASSERT_EQUAL_INT(2, output->cols);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(output, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(output, 0, 1));

  sm_destroy(output);
  // input, weights, bias are static and not freed here
}

void test_dense_backward_should_update_weights_and_bias(void) {
  float in_vals[1][2] = {{1.0f, 2.0f}};
  float w_vals[2][2] = {{0.5f, 0.0f}, {0.0f, 0.5f}};
  float b_vals[1][2] = {{0.0f, 0.0f}};
  float act_vals[1][2] = {{0.5f, 1.0f}};
  float grad_vals[1][2] = {{1.0f, 2.0f}};

  FloatMatrix *input = sm_from_array_static(1, 2, in_vals);
  FloatMatrix *weights = sm_from_array_static(2, 2, w_vals);
  FloatMatrix *bias = sm_from_array_static(1, 2, b_vals);
  FloatMatrix *activation = sm_from_array_static(1, 2, act_vals);
  FloatMatrix *grad = sm_from_array_static(1, 2, grad_vals);

  DenseLayer layer = {.weights = weights,
                      .bias = bias,
                      .activation = nm_apply_relu,
                      .activation_derivative = nm_d_relu};

  dense_backward(&layer, input, activation, grad, 0.1f);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.5f - 0.1f * 1.0f,
                           sm_get(weights, 0, 0)); // 0.4
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f - 0.1f * 2.0f,
                           sm_get(weights, 0, 1)); // -0.2
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f - 0.1f * 2.0f,
                           sm_get(weights, 1, 0)); // -0.2
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.5f - 0.1f * 4.0f,
                           sm_get(weights, 1, 1)); // 0.1
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, -0.1f * 1.0f, sm_get(bias, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, -0.1f * 2.0f, sm_get(bias, 0, 1));
}

void test_nm_sum_rows_should_sum_across_rows(void) {
  float values[3][4] = {{1.0f, 2.0f, 3.0f, 4.0f},
                        {5.0f, 6.0f, 7.0f, 8.0f},
                        {9.0f, 10.0f, 11.0f, 12.0f}};

  FloatMatrix *mat = sm_from_array_static(3, 4, values);
  FloatMatrix *sum = nm_sum_rows(mat);

  TEST_ASSERT_NOT_NULL(sum);
  TEST_ASSERT_EQUAL_UINT32(1, sum->rows);
  TEST_ASSERT_EQUAL_UINT32(4, sum->cols);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 15.0f, sm_get(sum, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 18.0f, sm_get(sum, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 21.0f, sm_get(sum, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 24.0f, sm_get(sum, 0, 3));

  sm_destroy(sum);
  // static matrix 'mat' needs no destruction
}

void test_train_one_epoch_should_reduce_loss_on_xor_data(void) {
  // XOR-Eingaben (4 Beispiele, 2 Features)
  float x_vals[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  // Zielwerte (One-Hot): 0 => [1,0], 1 => [0,1]
  float y_vals[4][2] = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};

  FloatMatrix *X = sm_from_array_static(4, 2, x_vals);
  FloatMatrix *Y = sm_from_array_static(4, 2, y_vals);

  // Einfaches 2-2-2 Netz
  DenseLayer layers[2];
  layers[0].weights = sm_create_random_xavier(2, 2, 2, 2);
  layers[0].bias = sm_create_zeros(1, 2);
  layers[0].activation = nm_apply_sigmoid;
  layers[0].activation_derivative = nm_d_sigmoid;

  layers[1].weights = sm_create_random_xavier(2, 2, 2, 2);
  layers[1].bias = sm_create_zeros(1, 2);
  layers[1].activation = nm_apply_softmax;
  layers[1].activation_derivative = NULL; // handled separately in training

  NeuralNetwork net = {.layers = layers, .num_layers = 2};

  // Vorher: Loss messen
  FloatMatrix *out_before = dense_forward(&layers[0], X);
  FloatMatrix *final_before = dense_forward(&layers[1], out_before);
  float loss_before = nm_cross_entropy_loss(final_before, Y);
  sm_destroy(out_before);
  sm_destroy(final_before);

  // 10 Epochen trainieren
  for (int epoch = 0; epoch < 10; ++epoch) {
    train_one_epoch(&net, X, Y, 4, 0.5f);
  }

  // Nachher: Loss erneut messen
  FloatMatrix *out_after = dense_forward(&layers[0], X);
  FloatMatrix *final_after = dense_forward(&layers[1], out_after);
  float loss_after = nm_cross_entropy_loss(final_after, Y);
  sm_destroy(out_after);
  sm_destroy(final_after);

  // Test: Verlust sollte gesunken sein
  TEST_ASSERT_TRUE(loss_after < loss_before);

  sm_destroy(layers[0].weights);
  sm_destroy(layers[0].bias);
  sm_destroy(layers[1].weights);
  sm_destroy(layers[1].bias);
}
//

void test_predict_should_return_output_for_forward_pass(void) {
  float input_vals[1][2] = {{1.0f, 2.0f}};
  FloatMatrix *input = sm_from_array_static(1, 2, input_vals);

  // Set up simple network with 1 layer
  DenseLayer layer;
  float w_vals[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
  float b_vals[1][2] = {{0.0f, 0.0f}};
  layer.weights = sm_from_array_static(2, 2, w_vals);
  layer.bias = sm_from_array_static(1, 2, b_vals);
  layer.activation = nm_apply_relu;
  layer.activation_derivative = nm_d_relu;

  NeuralNetwork net = {.layers = &layer, .num_layers = 1};

  FloatMatrix *output = predict(&net, input);

  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_EQUAL_UINT32(1, output->rows);
  TEST_ASSERT_EQUAL_UINT32(2, output->cols);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(output, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(output, 0, 1));

  sm_destroy(output);
}

void test_nm_argmax_rowwise_should_return_index_of_max_per_row(void) {
  float values[3][4] = {
    {0.1f, 0.9f, 0.3f, 0.7f},  // max at index 1
    {0.0f, 0.0f, 1.0f, 0.0f},  // max at index 2
    {5.0f, 2.0f, 4.0f, 8.0f}   // max at index 3
  };

  FloatMatrix *mat = sm_from_array_static(3, 4, values);
  FloatMatrix *indices = nm_argmax_rowwise(mat);

  TEST_ASSERT_NOT_NULL(indices);
  TEST_ASSERT_EQUAL_UINT32(3, indices->rows);
  TEST_ASSERT_EQUAL_UINT32(1, indices->cols);

  TEST_ASSERT_EQUAL_FLOAT(1.0f, sm_get(indices, 0, 0));
  TEST_ASSERT_EQUAL_FLOAT(2.0f, sm_get(indices, 1, 0));
  TEST_ASSERT_EQUAL_FLOAT(3.0f, sm_get(indices, 2, 0));

  sm_destroy(indices);
}