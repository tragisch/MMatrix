#include "armstrong_numbers.h"

bool is_armstrong_number(int candidate) {
  int r, sum = 0, temp;

  temp = candidate;

  while (candidate > 0) {
    r = candidate % 10;
    sum = sum + (r * r * r);
    candidate = candidate / 10;
  }

  if (temp == sum) {
    return true;
  }

  return false;
}
