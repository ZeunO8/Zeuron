# nn++

A simple NeuralNetwork library written in C++

Uses CMake for it's build system and comes with some included tests

### Building

Either use your preferred IDE of choice, or run the following commands to build

```bash
cmake -B build .
cmake --build build
```

### Testing

```bash
ctest --test-dir build --rerun-failed --output-on-failure -C Debug
```

### Usage

```cpp
#include <NeuralNetwork.hpp>
#include <Logger.hpp>
// Create a Neural Network like so
std::shared_ptr<NeuralNetwork> neuralNetworkPointer(
    new NeuralNetwork(
        // Layer sizes
        std::vector<unsigned long>({ 1, 10, 20, 10, 1 }),
        // ActivationType. Can be one of: Sigmoid, Linear, Swish, Tanh
        NeuralNetwork::Tanh
    )
);
auto &network = *neuralNetworkPointer;
// Training
// Declare some training inputs/outputs
// Simple XOR example
std::vector<std::vector<long double>> trainingInputs = {{{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}};
std::vector<std::vector<long double>> trainingOutputs = {{{{0}}, {{1}}, {{1}}, {{0}}}};
// Train
// Set learning rate
network.learningRate = 20;
// Train a number of iterations
auto trainingInputsSize = trainingInputs.size();
unsigned long trainingIteration = 0;
for (; trainingIteration < 4096; trainingIteration++)
{
    for (unsigned long trainingIndex = 0; trainingIndex < trainingInputsSize; trainingIndex++)
    {
        auto &input = trainingInputs[trainingIndex];
        auto &output = trainingOutputs[trainingIndex];
        // Activate the network with input
        network.feedforward(input);
        // Backpropagate the network with expected output
        network.backpropagate(output);
    }
}
// Use the network
std::vector<long double> input({ {0, 1} });
network.feedforward(input);
auto outputs = network.getOutputs();
logger(Logger::Info, "Output: " + std::to_string(outputs[0]));
```