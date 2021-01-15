#include <fstream>
#include <iostream>

#include "dataset.h"
#include "model.h"
#include "trainingTensor.h"

constexpr unsigned int EPOCHS = 100;
constexpr unsigned int BATCH_SIZE = 8;
constexpr unsigned int BATCHES_PER_EPOCH = 100;
constexpr unsigned int SAMPLES_PER_EPOCH = BATCH_SIZE * BATCHES_PER_EPOCH;
constexpr unsigned int SAVE_EVERY = 1;

void saveResult(TrainingTensor const input, torch::Tensor const &output, std::string const &path, unsigned int index = -1)
{
  cimg_library::CImg<float> output_image(output.sizes()[3], output.sizes()[2], 1, 3, 0);
  cimg_forXY(output_image, x, y)
  {
    output_image(x, y, 0, 0) = output[0][0][y][x].item<float>() * 255;
    output_image(x, y, 0, 1) = output[0][0][y][x].item<float>() * 255;
    output_image(x, y, 0, 2) = output[0][0][y][x].item<float>() * 255;
  }
  cimg_library::CImg<float> input_image(output.sizes()[3], output.sizes()[2], 1, 3, 0);
  cimg_forXY(input_image, x, y)
  {
    input_image(x, y, 0, 0) = input._data[0][0][y][x].item<float>() * 255;
    input_image(x, y, 0, 1) = input._data[0][1][y][x].item<float>() * 255;
    input_image(x, y, 0, 2) = input._data[0][2][y][x].item<float>() * 255;
  }
  cimg_library::CImg<float> label_image(output.sizes()[3], output.sizes()[2], 1, 3, 0);
  cimg_forXY(label_image, x, y)
  {
    label_image(x, y, 0, 0) = input._labels[0][0][y][x].item<float>() * 255;
    label_image(x, y, 0, 1) = input._labels[0][0][y][x].item<float>() * 255;
    label_image(x, y, 0, 2) = input._labels[0][0][y][x].item<float>() * 255;
  }

  output_image.append(input_image, 'y');
  output_image.append(label_image, 'y');
  output_image.save(path.c_str(), index);
}

float train(PointExtractor &model, SyntheticShapeDataset &dataset, torch::optim::Adam &optimizer, unsigned int epoch)
{
  auto data_loader = torch::data::make_data_loader(dataset, torch::data::samplers::StreamSampler(SAMPLES_PER_EPOCH), BATCH_SIZE);
  float epoch_loss = 0.0f;
  unsigned int i(0);
  for (TrainingTensor batch : *data_loader)
  {
    optimizer.zero_grad();
    torch::Tensor prediction = model->forward(batch._data);
    if (!i)
      saveResult(batch, prediction, "output.png", epoch);
    torch::Tensor loss = torch::nn::functional::mse_loss(prediction, batch._labels, torch::nn::MSELossOptions(torch::kSum));
    epoch_loss += loss.item<float>();
    loss.backward();
    optimizer.step();
    i++;
  }
  return epoch_loss;
}

int main(int ac, char **av)
{
  torch::Device device(torch::kCUDA);
  SyntheticShapeDataset dataset(device);
  PointExtractor model(32, 1);
  model->to(device);
  torch::optim::Adam optimizer(model->parameters(), 1e-4);
  model->train();
  std::cout << model << std::endl;

  std::ofstream train_loss_file("train.txt");
  for (unsigned int epoch(0); epoch < EPOCHS; epoch++)
  {
    float epoch_loss = train(model, dataset, optimizer, epoch);
    // No evaluation method as the dataset is generated on the fly.
    std::cout << "Epoch Loss : " << epoch_loss << std::endl;
    train_loss_file << epoch_loss << std::endl;
    if (epoch % SAVE_EVERY == 0)
      torch::save(model, "point_extractor.pt");
  }

  return 0;
}
