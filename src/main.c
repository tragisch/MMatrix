#include <stdio.h>

#include "dm.h"
#include "dm_io.h"

int main(void) {
  // setup random sparse matrix
  // DoubleSparseMatrix *sparse_matrix = dms_rand(1000, 1000, 0.005);
  // dms_cplot(sparse_matrix, 1.0);
  // dms_destroy(sparse_matrix);

  DoubleSparseMatrix *sparse_matrix2 = dms_identity(100);
  dms_cplot(sparse_matrix2, 5.0);
  dms_destroy(sparse_matrix2);
}