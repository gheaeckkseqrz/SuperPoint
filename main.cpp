#include <iostream>

#include "dataset.h"
#include "model.h"
#include "trainingTensor.h"

constexpr unsigned int EPOCHS = 100;
constexpr unsigned int BATCH_SIZE = 8;
constexpr unsigned int BATCHES_PER_EPOCH = 1000;
constexpr unsigned int SAMPLES_PER_EPOCH = BATCH_SIZE * BATCHES_PER_EPOCH;

float train(PointExtractor &model, SyntheticShapeDataset &dataset, torch::optim::Adam &optimizer, bool train)
{
  auto data_loader = torch::data::make_data_loader(dataset, torch::data::samplers::StreamSampler(SAMPLES_PER_EPOCH), BATCH_SIZE);
  for (TrainingTensor batch : *data_loader)
  {
    std::cout << batch._data.sizes() << std::endl;
    std::cout << batch._labels.sizes() << std::endl;
    break;
  }
  return 0.0f;
}

int main(int ac, char **av)
{
  SyntheticShapeDataset dataset;
  PointExtractor model(32, 1);
  torch::optim::Adam optimizer(model->parameters(), 1e-4);
  std::cout << model << std::endl;

  train(model, dataset, optimizer, true);

  return 0;
}
