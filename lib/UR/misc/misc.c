#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// enforce order of includes
#include "misc.h"

/**********************
 *** STRINGS
 ************************/

char *file2string(char *path) {
  FILE *fp;
  fp = fopen(path, "r");
  char *buffer = 0;
  long length;

  if (fp) {
    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buffer = malloc(length);
    if (buffer) {
      fread(buffer, 1, length, fp);
    }
    fclose(fp);
  } else {
    printf("no file found, or can't read.");
  }
  return buffer;
}

int string2file(char *filepath, char *data) {
  int rc = 0;

  FILE *fOut = fopen(filepath, "ab+");
  if (fOut != NULL) {
    if (fputs(data, fOut) != EOF) {
      rc = 1;
    }
    if (fclose(fOut) == EOF) rc = 0;
  }

  return rc;
}

void slice(const char *str, char *buffer, size_t start, size_t end) {
  size_t j = 0;
  for (size_t i = start; i <= end; ++i) {
    buffer[j++] = str[i];
  }
  buffer[j] = 0;
}

// error checking omitted for sake of brevity
myString *mystring_init(const char *src) {
  myString *pms = malloc(sizeof(myString));
  pms->len = strlen(src + 1);
  pms->capacity = pms->len;
  pms->str = (char *)malloc(pms->capacity + 1);
  strcpy(pms->str, src);
  return pms;
}

// error checking omitted for sake of brevity
int mystring_cat(myString *pms, const char *src) {
  // if (pms == NULL) return 1;
  pms->len += strlen(src + 1);
  pms->capacity = pms->len;
  pms->str = (char *)realloc(pms->str, pms->capacity + 1);
  strcat(pms->str, src);
  return 0;
}

// destroy pointer:
int mystring_destroy(myString *pms) {
  if (pms == NULL) return 1;
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
double randomDouble() { return (double)arc4random() / (double)RAND_MAX; }

/**
 * @brief returns a random 32-bit unsigned integer.
 *
 * @return uint
 */
uint32_t randomInt() { return arc4random(); }
uint32_t randomInt_upperBound(uint32_t limit) {
  return arc4random_uniform(limit);
}
uint32_t randomInt_betweenBounds(uint32_t min, uint32_t max) {
  if (max < min) return min;
  return arc4random_uniform((max - min) + 1) + min;
}
