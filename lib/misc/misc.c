// enforce order of includes
#include "misc.h"

#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4
#define UNITY_DOUBLE_PRECISION 0.0000001

/**********************
 *** STRINGS
 ************************/

char *file2string(char *path) {
  FILE *file = fopen(path, "r");
  char *buffer = 0;

  if (file) {
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    buffer = malloc(length);
    if (buffer) {
      fread(buffer, 1, length, file);
    }
    fclose(file);
  } else {
    printf("no file found, or can't read.");
  }
  return buffer;
}

int string2file(char *filepath, char *data) {
  int write_ok = 0;

  FILE *fOut = fopen(filepath, "ab+");
  if (fOut != NULL) {
    if (fputs(data, fOut) != EOF) {
      write_ok = 1;
    }
    if (fclose(fOut) == EOF) {
      write_ok = 0;
    }
  }

  return write_ok;
}

void slice(const char *str, char *buffer, size_t start, size_t end) {
  size_t idx = 0;
  for (size_t i = start; i <= end; ++i) {
    buffer[idx++] = str[i];
  }
  buffer[idx] = 0;
}

// error checking omitted for sake of brevity
myString *mystring_init(const char *src) {
  myString *pms = malloc(sizeof(myString));
  pms->len = (int)strlen(src + 1);
  pms->capacity = pms->len;
  pms->str = (char *)malloc(pms->capacity + 1);
  strcpy(pms->str, src);
  return pms;
}

// error checking omitted for sake of brevity
int mystring_cat(myString *pms, const char *src) {
  // if (pms == NULL) return 1;
  pms->len += (int)strlen(src + 1);
  pms->capacity = pms->len;
  pms->str = (char *)realloc(pms->str, pms->capacity + 1);
  strcat(pms->str, src);
  return 0;
}

// destroy pointer:
int mystring_destroy(myString *pms) {
  if (pms == NULL) {
    return 1;
  }
  pms->str = NULL;
  free(pms->str);
  pms = NULL;
  free(pms);
  return 0;
}

/*******************************/
/*   Random functions          */
/*******************************/

/**
 * @brief return random number between 0 ... 1
 *
 * @return double
 */
double randomDouble() {
  uint32_t random_uint32 = randomInt();
  double random_double = (double)random_uint32 / (double)UINT32_MAX;
  return random_double;
}
double randomDouble_betweenBounds(uint32_t min, uint32_t max) {
  return (randomInt_betweenBounds(min, max - 1) + randomDouble());
}

/**
 * @brief returns a random 32-bit unsigned integer.
 *
 * @return uint
 */
uint32_t randomInt() { return random_number_generator(); }

#ifdef __APPLE__

uint32_t randomInt_upperBound(uint32_t limit) {
  return arc4random_uniform(limit);
}

#else

uint32_t randomInt_upperBound(uint32_t limit) {
  static int initialized = 0;
  if (!initialized) {
    srand(time(NULL));
    initialized = 1;
  }
  int r;
  do {
    r = rand();
  } while (r >= RAND_MAX - RAND_MAX % limit);
  return r % limit;
}

#endif

uint32_t randomInt_betweenBounds(uint32_t min, uint32_t max) {
  if (max < min) {
    return min;
  }
  return randomInt_upperBound((max - min) + 1) + min;
}

/*******************************/
/*        Double Array       */
/*******************************/

/**
 * @brief Create a DoubleArray object in HEAP
 *
 * @param length
 * @param value
 * @return double*
 */
double *createRandomDoubleArray(unsigned int length) {
  double *array = (double *)calloc(length, sizeof(double));

  for (unsigned int i = 0; i < length; i++) {
    array[i] = randomDouble();
  }
  return array; // die Speicheradresse wird zurückgegeben.
}
