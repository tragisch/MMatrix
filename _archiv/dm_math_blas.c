



// // use cblas_dgemm to multiply two dense matrices
// static DoubleMatrix *dm_multiply_by_matrix_blas(const DoubleMatrix *mat1,
//                                                 const DoubleMatrix *mat2) {

// #ifdef APPLE_BLAS
//   printf("Using Apple's Accelerate.h'\n");
// #else
//   printf("Using OpenBLAS\n");
// #endif // APPLE_BLAS

//   DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);

//   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (BLASINT)mat1->rows,
//               (BLASINT)mat2->cols, (BLASINT)mat1->cols, 1.0, mat1->values,
//               (BLASINT)mat1->cols, mat2->values, (BLASINT)mat2->cols, 0.0,
//               product->values, (BLASINT)product->cols);

//   return product;
// }

