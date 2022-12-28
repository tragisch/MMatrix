
/*
 * Examples of gnuplot_i.c usage
 */

#include <stdio.h>
#include <stdlib.h>

#include "gnuplot_i.h"

#define SLEEP_LGTH 1

int main(int argc, char *argv[]) {
  gnuplot_ctrl *h1;
  double phase;

  printf("*** example of gnuplot control through C ***\n");
  h1 = gnuplot_init();

  for (int i = 10; i < 100; i += 10) {
    phase = i / 10;
    gnuplot_resetplot(h1);
    gnuplot_cmd(h1, "plot sin(x+%g)", phase);
  }

  for (int i = 100; i >= 10; i -= 10) {
    phase = i / 10;
    gnuplot_resetplot(h1);
    gnuplot_cmd(h1, "plot sin(x+%g)", phase);
  }

  gnuplot_close(h1);
  return 0;
}
