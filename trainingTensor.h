#pragma once

#include <torch/torch.h>

#include "trainingImage.h"

struct TrainingTensor
{
  TrainingTensor(unsigned int batch_size, unsigned int w, unsigned int h)
  {
    _data = torch::zeros({batch_size, 3, h, w}, torch::TensorOptions().requires_grad(false));
    _labels = torch::zeros({batch_size, 1, h, w}, torch::TensorOptions().requires_grad(false));
  }

  TrainingTensor(TrainingImage const &image)
  {
    _data = torch::from_blob(const_cast<float *>(image._img.data()), {3, image._img.width(), image._img.height()});
    _data = _data.clone(); // Clone the memory, so that image can be released
    _data /= 255;          // Normalize to [0, 1]

    _labels = torch::zeros({1, image._img.width(), image._img.height()});
    createLabel(image._keypoints, _labels);
  }

  static void createLabel(std::vector<Vec2> const &keypoints, torch::Tensor &label)
  {
    label.fill_(0);
    for (Vec2 const &keypoint : keypoints)
      label[0][keypoint.y][keypoint.x] = 1;
  }

  torch::Tensor _data;
  torch::Tensor _labels;
};
