#include <stdio.h>

#include "misc.h"

int main() {
  // initialise mytring
  char *str = "mein erster String";
  myString *mstr = mystring_init(str);

  // so anything:
  mystring_cat(mstr, " und ein weiterer Teil.");
  printf("%s\n", mstr->str);

  // destroy mystring:
  mystring_destroy(mstr);

  // random number
  for (size_t i = 0; i < 100; i++) {
    printf("Dice: %d\n", randomInt_upperBound(6));
    printf("Double: %f\n", randomDouble());
  }
  return 0;
}
