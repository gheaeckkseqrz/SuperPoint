#pragma once

#include "trainingImage.h"

class SyntheticShapeGenerator
{
public:
  SyntheticShapeGenerator() = default;

  TrainingImage generateTrainingImage(unsigned int w = 256, unsigned int h = 256) const;

private:
  void generateBackground(TrainingImage &img) const;
  void drawTriangles(TrainingImage &img) const;
  void drawTriangle(TrainingImage &img) const;
};
