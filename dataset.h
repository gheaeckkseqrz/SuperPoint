#pragma once

#include <torch/torch.h>

#include "imageGenerator.h"
#include "trainingTensor.h"

class SyntheticShapeDataset : protected SyntheticShapeGenerator, public torch::data::datasets::StreamDataset<SyntheticShapeDataset, TrainingTensor>
{
public:
  SyntheticShapeDataset(torch::Device device)
    : _device(device)
  {
  }

  TrainingTensor get_batch(size_t batch_size) override
  {
    TrainingTensor batch(batch_size, _w, _h);
    for (unsigned int i(0); i < batch_size; ++i)
    {
      TrainingImage image = generateTrainingImage(_w, _h);
      TrainingTensor tensor(image);
      batch._data[i].copy_(tensor._data[0]);
      batch._labels[i].copy_(tensor._labels[0]);
    }
    return batch.to(_device);
  }

  c10::optional<size_t> size() const override
  {
    return torch::nullopt;
  }

private:
  torch::Device _device;
  unsigned int _w = 256;
  unsigned int _h = 256;
};
