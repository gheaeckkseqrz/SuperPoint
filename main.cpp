#include <iostream>

#include "dataset.h"

int main(int ac, char **av)
{
  SyntheticShapeDataset dataset;

  for (unsigned int i(0); i < 10; ++i)
  {
    TrainingImage img = dataset.generateTrainingImage();
    img._img.save("training_image.png", i);
  }

  return 0;
}
