#include <torch/torch.h>

#include <vector>

at::Tensor relu(at::Tensor z, at::Scalar alpha = 1.0) {
  
  return (z > 0).type_as(z);
}

std::vector<at::Tensor> nnan_forward(
    at::Tensor inputs;
    at::Tensor orig_shape;
    at::Tensor outputs;
    
