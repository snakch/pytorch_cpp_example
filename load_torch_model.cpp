#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::kCUDA;
    } else
        std::cout << "CUDA not available! Training on CPU." << std::endl;

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs_ones;
    inputs_ones.push_back(torch::ones({5}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output_ones = module.forward(inputs_ones).toTensor();
    std::cout << "Pushing ones through resnet...in C++! \n";
    std::cout << output_ones.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

    // Same again with zeros
    std::vector<torch::jit::IValue> inputs_zeros;
    inputs_zeros.push_back(torch::zeros({5}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output_zeros = module.forward(inputs_zeros).toTensor();
    std::cout << "Pushing zeros through resnet...in C++! \n";
    std::cout << output_zeros.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';

    // Put the model (and inputs) on the gpu

    module.to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs_ones_gpu;
    inputs_ones_gpu.push_back(torch::ones({5}).to(at ::kCUDA));

    at::Tensor output_ones_gpu = module.forward(inputs_ones_gpu).toTensor();
    std::cout << "Pushing ones through resnet...in C++, on the gpu! \n";
    std::cout << output_ones_gpu.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';
}
