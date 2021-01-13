#include <iostream>

#include "dataset.h"
#include "model.h"
#include "trainingTensor.h"

constexpr unsigned int EPOCHS = 100;
constexpr unsigned int BATCH_SIZE = 8;
constexpr unsigned int BATCHES_PER_EPOCH = 100;
constexpr unsigned int SAMPLES_PER_EPOCH = BATCH_SIZE * BATCHES_PER_EPOCH;
constexpr unsigned int SAVE_EVERY = 1;

float train(PointExtractor &model, SyntheticShapeDataset &dataset, torch::optim::Adam &optimizer)
{
  auto data_loader = torch::data::make_data_loader(dataset, torch::data::samplers::StreamSampler(SAMPLES_PER_EPOCH), BATCH_SIZE);
  float epoch_loss = 0.0f;
  for (TrainingTensor batch : *data_loader)
  {
    optimizer.zero_grad();
    torch::Tensor prediction = model->forward(batch._data);
    torch::Tensor loss = torch::nn::functional::mse_loss(prediction, batch._labels);
    epoch_loss += loss.item<float>();
    loss.backward();
    optimizer.step();
  }
  return epoch_loss;
}

int main(int ac, char **av)
{
  SyntheticShapeDataset dataset;
  PointExtractor model(32, 1);
  torch::optim::Adam optimizer(model->parameters(), 1e-4);
  model->train();
  std::cout << model << std::endl;

  for (unsigned int epoch(0); epoch < EPOCHS; epoch++)
  {
    float epoch_loss = train(model, dataset, optimizer);
    // No evaluation method as the dataset is generated on the fly.
    std::cout << "Epoch Loss : " << epoch_loss << std::endl;
    if (epoch % SAVE_EVERY == 0)
      torch::save(model, "point_extractor.pt");
  }

  return 0;
}
