#include "pbPlots.h"
#include "supportLib.h"

#define points 50

int main() {
  _Bool success;
  double x[points];
  double y[points];

  for (int i = 0; i < points; i++) {
    x[i] = rand();
    y[i] = rand();
  }

  double minX = 0.;
  double maxX = 1.;
  double minY = 0.;
  double maxY = 1.;

  RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
  StringReference *errorMessage;
  errorMessage = (StringReference *)malloc(sizeof(StringReference));

  ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
  settings->width = 800;
  settings->height = 600;
  settings->autoBoundaries = false;
  settings->xMax = maxX;
  settings->xMin = minX;
  settings->yMax = maxY;
  settings->yMin = minY;

  success =
      DrawScatterPlot(imageRef, 800, 600, x, points, y, points, errorMessage);

  if (success) {
    size_t length;
    double *pngdata = ConvertToPNG(&length, imageRef->image);
    DeleteImage(imageRef->image);

    WriteToFile(pngdata, length, "example3.png");
  } else {
    fprintf(stderr, "Error: ");
    for (int i = 0; i < errorMessage->stringLength; i++) {
      fprintf(stderr, "%c", errorMessage->string[i]);
    }
    fprintf(stderr, "\n");
  }

  return success ? 0 : 1;
}
