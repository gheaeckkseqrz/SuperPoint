#pragma once

#include <torch/torch.h>

#include "trainingImage.h"

struct TrainingTensor
{
  TrainingTensor(TrainingImage const &image)
    : _keypoints(image._keypoints)
  {
    _t = torch::from_blob(const_cast<float *>(image._img.data()), {3, image._img.width(), image._img.height()});
    _t = _t.clone(); // Clone the memory, so that image can be released
    _t /= 255;       // Normalize to [0, 1]
  }

  torch::Tensor _t;
  std::vector<Vec2> _keypoints;
};
