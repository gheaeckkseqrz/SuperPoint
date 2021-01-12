#pragma once

#include <torch/torch.h>

#include "imageGenerator.h"
#include "trainingTensor.h"

class SyntheticShapeDataset : protected SyntheticShapeGenerator, public torch::data::datasets::StreamDataset<SyntheticShapeDataset, TrainingTensor>
{
public:
  TrainingTensor get_batch(size_t batch_size) override
  {
    TrainingImage image = generateTrainingImage(256, 256);
    return TrainingTensor(image);
  }

  c10::optional<size_t> size() const override
  {
    return torch::nullopt;
  }
};
