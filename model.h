#pragma once

#include <torch/torch.h>

class PointExtractorImpl : public torch::nn::Module
{
public:
  PointExtractorImpl(unsigned int nc, unsigned int nz)
  {
    _conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, nc, 3).padding(1).stride(1).bias(true)));
    _conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).stride(1).bias(true)));
    _conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nc, 3).padding(1).stride(1).bias(true)));
    _conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, nz, 3).padding(1).stride(1).bias(true)));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::nn::functional::relu(_conv1(x));
    x = torch::nn::functional::relu(_conv2(x));
    x = torch::nn::functional::relu(_conv3(x));
    x = _conv4(x);
    return x;
  }

private:
  torch::nn::Conv2d _conv1 = nullptr;
  torch::nn::Conv2d _conv2 = nullptr;
  torch::nn::Conv2d _conv3 = nullptr;
  torch::nn::Conv2d _conv4 = nullptr;
};

TORCH_MODULE(PointExtractor);
