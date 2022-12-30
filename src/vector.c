/**
 * @file vector.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "vector.h"

/*******************************/
/*         I/O Functions       */
/*******************************/

/* read in DoubleVector data from file */
int readInDoubleVectorData(DoubleVector* vec, const char* filepath) {
  FILE* fp = fopen(filepath, "r");
  if (fp == NULL) {
    return 1;
  }
  int succ_read = 1;
  for (size_t i = 0; i < vec->length; i++) {
    // FIXME: insecure use of fscanf:
    succ_read = fscanf(fp, "%lf", &vec->double_array[i]);
  }
  fclose(fp);

  return succ_read;
}

/* write data from DoubleVector to file */
int writeOutDoubleVectorData(DoubleVector* vec, const char* filepath) {
  FILE* fp = fopen(filepath, "w");
  if (fp == NULL) {
    return 1;
  }

  for (size_t i = 0; i < vec->length; i++) {
    if (i < vec->length - 1) {
      fprintf(fp, "%lf\n", vec->double_array[i]);
    } else {
      fprintf(fp, "%lf", vec->double_array[i]);
    }
  }
  fclose(fp);

  return 0;
}

/*******************************/
/*  Double Vector (Dynamic)    */
/*******************************/

/**
 * @brief Create a DoubleVector object (HEAP INIT_CAPACITY)
 * @return DoubleVector*
 */

DoubleVector* createDoubleVector() {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  if (!vec) return NULL;

  vec->length = 0u;
  vec->capacity = INIT_CAPACITY;

  double* array = (double*)malloc(vec->capacity * sizeof(double));
  vec->double_array = array;

  return vec;
}

/**
 * @brief Clone a DoubleVector object
 * @return DoubleVector*
 */
DoubleVector* cloneDoubleVector(const DoubleVector* vector) {
  size_t org_length = vector->length;
  DoubleVector* clone = createDoubleVectorOfLength(org_length, 0.);
  for (size_t i = 0; i < org_length; i++) {
    clone->double_array[i] = vector->double_array[i];
  }

  return clone;
}

/**
 * @brief Create a Double Vector Of Length object
 *
 * @param length
 * @param value
 * @return DoubleVector*
 */
DoubleVector* createDoubleVectorOfLength(size_t length, double value) {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  double* array = (double*)malloc(length * sizeof(double));

  for (size_t i = 0; i < length; i++) {
    array[i] = value;
  }

  vec->double_array = array;
  vec->length = length;
  if (vec->length > INIT_CAPACITY) {
    vec->capacity = length + INIT_CAPACITY;
  } else {
    vec->capacity = INIT_CAPACITY;
  }
  return vec;
}

/**
 * @brief Create a Random Double Vector object
 *
 * @param length
 * @return DoubleVector
 */
DoubleVector* createRandomDoubleVectorOfLength(size_t length) {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  double* array = (double*)malloc(length * sizeof(double));

  for (size_t i = 0; i < length; i++) {
    array[i] = randomDouble();
  }

  vec->double_array = array;
  vec->length = length;
  if (vec->length > INIT_CAPACITY) {
    vec->capacity = length + INIT_CAPACITY;
  } else {
    vec->capacity = INIT_CAPACITY;
  }
  return vec;
}

/**
 * @brief expand allocate memory in HEAP with another INIT_CAPACITY
 *
 * @param vec
 */
void expandDoubleVector(DoubleVector* vec) {
  vec->capacity += INIT_CAPACITY;  // TODO: or += vec->capacity;
  vec->double_array =
      realloc(vec->double_array, vec->capacity * sizeof(double));
}

/**
 * @brief shrink allocate memory in HEAP
 *
 * @param vec
 */
void shrinkDoubleVector(DoubleVector* vec) {  // l= 10 // 10
  if ((vec->length >= (vec->capacity - INIT_CAPACITY) &&
       (vec->capacity - INIT_CAPACITY) > 0)) {
    vec->capacity = (vec->capacity - INIT_CAPACITY);

    vec->double_array =
        realloc(vec->double_array, vec->capacity * sizeof(double));
  }
}

/**
 * @brief push (add) new value to vector vec
 *
 * @param vec
 * @param value
 */
void pushValue(DoubleVector* vec, double value) {
  if (vec->length == vec->capacity) {
    expandDoubleVector(vec);
  }
  vec->double_array[vec->length] = value;
  vec->length++;
}

/**
 * @brief pop (get) last element if DoubleVector vec
 *
 * @param vec
 * @return double
 */
double popValue(DoubleVector* vec) {
  double value = vec->double_array[vec->length - 1];

  vec->double_array[vec->length] = 0.0;
  vec->length--;

  if (vec->length < vec->capacity) {
    shrinkDoubleVector(vec);
  }

  return value;
}

/**
 * @brief printf DoubleArray pretty
 *
 * @param DoubleVector* vec
 */
void printDoubleVector(DoubleVector* vec) {
  double* array = vec->double_array;
  size_t length = vec->length;
  for (size_t i = 0; i < length; i++) {
    if (i == 0) {
      printf("[%.2lf ", array[i]);
    } else if (i == length - 1) {
      printf("%.2lf]\n", array[i]);
    } else {
      if (length < MAX_COLUMN) {
        printf("%.2lf ", array[i]);
      } else {
        if (i < MAX_COLUMN_PRINT) {
          printf("%.2lf ", array[i]);
        } else if (i == MAX_COLUMN_PRINT) {
          printf(" ... ");
        } else if (i > length - MAX_COLUMN_PRINT - 1) {
          printf("%.2lf ", array[i]);
        }
      }
    }
  }
  printf("Vector 1x%zi, Capacity %zi\n", vec->length, vec->capacity);
}

/**
 * @brief free memory of DoubleVector
 *
 * @param vec
 * @return DoubleVector*
 */
void freeDoubleVector(DoubleVector* vec) {
  free(vec->double_array);
  vec->double_array = NULL;
  free(vec);
  vec = NULL;
}

/**
 * @brief Multiply Vector v1 with Vectot v2  -- dot product!
 *
 * @param vec1
 * @param vec2
 * @return double
 */
double multiplyDoubleVectors(DoubleVector* vec1, DoubleVector* vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors have not same length");
    return 0;
  }

  double sum;
  size_t i;
  sum = 0;
  for (i = 0; i < vec1->length; i++) {
    sum += vec1->double_array[i] * vec2->double_array[i];
  }
  return sum;
}

/**
 * @brief add Vector vec1 with Vector vec2
 *
 * @param vec1
 * @param vec2 (const)
 */
void addDoubleVector(DoubleVector* vec1, const DoubleVector* vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors are not same length");
  }
  for (size_t i = 0; i < vec1->length; i++) {
    vec1->double_array[i] += vec2->double_array[i];
  }
}

/**
 * @brief sub Vector vec1 from Vector vec2 (vec1 - vec2)
 *
 * @param vec1
 * @param vec2 (const)
 */
void subDoubleVector(DoubleVector* vec1, const DoubleVector* vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors are not same length");
  }

  for (size_t i = 0; i < vec1->length; i++) {
    vec1->double_array[i] -= vec2->double_array[i];
  }
}

/**
 * @brief multiply each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 */
void multiplyScalarToVector(DoubleVector* vec, const double scalar) {
  for (size_t i = 0; i < vec->length; i++) {
    vec->double_array[i] = vec->double_array[i] * scalar;
  }
}

/**
 * @brief divied each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 */
void divideScalarToVector(DoubleVector* vec, const double scalar) {
  for (size_t i = 0; i < vec->length; i++) {
    vec->double_array[i] = vec->double_array[i] / scalar;
  }
}

/**
 * @brief return mean of Vector vec
 *
 * @param vec
 * @return double
 */
double meanOfDoubleVector(DoubleVector* vec) {
  double mean = 0.0;
  for (size_t i = 0; i < vec->length; i++) {
    mean += vec->double_array[i];
  }

  return (mean / vec->length);
}

/**
 * @brief return min of Vector vec
 *
 * @param vec
 * @return double
 */
double minOfDoubleVector(DoubleVector* vec) {
  double min = vec->double_array[0];
  for (size_t i = 0; i < vec->length; i++) {
    if (min > vec->double_array[i]) min = vec->double_array[i];
  }
  return min;
}

/**
 * @brief return max of Vector vec
 *
 * @param vec
 * @return double
 */
double maxOfDoubleVector(DoubleVector* vec) {
  double max = vec->double_array[0];
  for (size_t i = 0; i < vec->length; i++) {
    if (max < vec->double_array[i]) max = vec->double_array[i];
  }
  return max;
}
