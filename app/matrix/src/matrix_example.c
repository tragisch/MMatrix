#include "sm.h"


int main() {
    // Example usage of the FloatMatrix structure
    size_t rows = 3;
    size_t cols = 3;
    FloatMatrix *matrix = sm_create(rows, cols);
    
    if (matrix) {
        // Fill the matrix with some values
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                matrix->values[i * cols + j] = (float)(i * cols + j);
            }
        }

        sm_print(matrix);
        sm_destroy(matrix);
    } else {
        fprintf(stderr, "Failed to create matrix.\n");
    }

    return 0;
}
