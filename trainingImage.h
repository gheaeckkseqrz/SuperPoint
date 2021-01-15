#pragma once

// This define disable the display functions in CImg
// (This is to avoid the dependency on XOrg libraries)
#define cimg_display 0

#include <CImg.h>
#include <string>
#include <vector>

#include "vec2.h"

struct TrainingImage
{
  TrainingImage(unsigned int w = 256, unsigned int h = 256)
    : _img(w, h, 1, 3, 0)
  {
  }

  cimg_library::CImg<float> generatePNG() const
  {
    cimg_library::CImg<float> img(_img);
    cimg_library::CImg<float> grayscale(_img);
    grayscale = grayscale.RGBtoYCbCr().channel(0);           // Take the luminance channel
    grayscale.append(grayscale, 'c').append(grayscale, 'c'); // Stack it to get RGB

    float red[3] = {255, 0, 0};
    // Annotate keypoints
    for (Vec2 const &point : _keypoints)
      grayscale.draw_circle(point.x, point.y, 5, red);

    // Concatenate images on top of each other and save
    img.append(grayscale, 'y');
    return img;
  }

  void savePNG(std::string const &path, int index = -1) const
  {
    cimg_library::CImg<float> img = generatePNG();
    img.save(path.c_str(), index);
  }

  cimg_library::CImg<float> _img;
  std::vector<Vec2> _keypoints;
};
