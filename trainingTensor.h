#pragma once

#include <torch/torch.h>

#include "trainingImage.h"

struct TrainingTensor
{
  TrainingTensor(TrainingImage const &image)
  {
    _t = torch::from_blob(const_cast<float *>(image._img.data()), {3, image._img.width(), image._img.height()});
    _t = _t.clone(); // Clone the memory, so that image can be released
    _t /= 255;       // Normalize to [0, 1]

    _keypoints = torch::zeros({static_cast<int64_t>(image._keypoints.size()), 2});
    for (std::size_t i(0); i < image._keypoints.size(); ++i)
    {
      _keypoints[i][0] = image._keypoints[i].x;
      _keypoints[i][1] = image._keypoints[i].y;
    }
  }

  torch::Tensor _t;
  torch::Tensor _keypoints;
};
