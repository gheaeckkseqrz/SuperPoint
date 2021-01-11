#pragma once

// This define disable the display functions in CImg
// (This is to avoid the dependency on XOrg libraries)
#define cimg_display 0

#include <CImg.h>
#include <vector>

#include "vec2.h"

struct TrainingImage
{
  TrainingImage(unsigned int w = 256, unsigned int h = 256)
    : _img(w, h, 1, 3, 0)
  {
  }

  cimg_library::CImg<float> _img;
  std::vector<Vec2> _keypoints;
};
