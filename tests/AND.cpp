/*
*/
#include <NeuralNetwork.hpp>
#include <Logger.hpp>
#include <memory>
#include <cassert>
using namespace nnpp;
/*
 */
int main()
{
	std::vector<std::vector<long double>> trainingInputs = {{{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}};
	std::vector<std::vector<long double>> trainingOutputs = {{{{0}}, {{0}}, {{0}}, {{1}}}};
	std::shared_ptr<NeuralNetwork> neuralNetworkPointer(
		new NeuralNetwork(
			2,
			{{NeuralNetwork::Sigmoid, 2}, {NeuralNetwork::Sigmoid, 1}}
		)
	);
	auto &network = *neuralNetworkPointer;
	auto trainingInputsSize = trainingInputs.size();
	network.learningRate = 20;
	unsigned long trainingIteration = 0;
	for (; trainingIteration < 4096; trainingIteration++)
	{
		for (unsigned long trainingIndex = 0; trainingIndex < trainingInputsSize; trainingIndex++)
		{
			auto &input = trainingInputs[trainingIndex];
			auto &output = trainingOutputs[trainingIndex];
			network.feedforward(input);
			network.backpropagate(output);
		}
	}
	logger(Logger::Info, "Trained " + std::to_string(trainingIteration) + " iterations");
	static const long double tolerance = 0.05;
	for (unsigned long trainingIndex = 0; trainingIndex < trainingInputsSize; trainingIndex++)
	{
		auto &input = trainingInputs[trainingIndex];
		auto &expectedOutput = trainingOutputs[trainingIndex];
		network.feedforward(input);
		auto actualOutputs = network.getOutputs();
		auto actualOutputsSize = actualOutputs.size();
		for (unsigned long outputIndex = 0; outputIndex < actualOutputsSize; ++outputIndex)
		{
			long double difference = std::abs(actualOutputs[outputIndex] - expectedOutput[outputIndex]);
			logger(Logger::Info,
				"For input { " + std::to_string(input[0]) +
						", " + std::to_string(input[1]) + " } the network has a difference of: " + std::to_string(difference) +
						", output: " + std::to_string(actualOutputs[outputIndex]) +
						", is " + (difference <= tolerance ? "within" : "not within") + " tolerance of " + std::to_string(tolerance));
		}
	}
	return 0;
};
/*
 */