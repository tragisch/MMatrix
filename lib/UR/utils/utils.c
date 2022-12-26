#include <stdio.h>
#include <stdlib.h>
// enforce order of includes
#include "utils.h"

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
