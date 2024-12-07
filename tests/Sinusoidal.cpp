/*
*/
#include <NeuralNetwork.hpp>
#include <Logger.hpp>
#include <memory>
#include <cassert>
#define _USE_MATH_DEFINES
#include <math.h>
#include <Visualizer.hpp>
using namespace nnpp;
/*
 * Sinusoidal Function
 * Predict y=sin(x) for x values in the range [0,π].
 */
int main()
{
	std::vector<std::vector<long double>> trainingInputs = {{{{0}}, {{0.1}}, {{0.2}}, {{0.3}}, {{0.4}}, {{0.5}}, {{0.6}}, {{0.7}}, {{0.8}}, {{0.9}}, {{1.0}}, {{1.1}}, {{1.2}}, {{1.3}}, {{1.4}}, {{1.5}}, {{1.6}}, {{1.7}}, {{1.8}}, {{1.9}}, {{2.0}}, {{2.1}}, {{2.2}}, {{2.3}}, {{2.4}}, {{2.5}}, {{2.6}}, {{2.7}}, {{2.8}}, {{2.9}}, {{3.0}}, {{3.1}}, {{3.2}}, {{3.3}}, {{3.4}}, {{3.5}}, {{3.6}}, {{3.7}}, {{3.8}}, {{3.9}}, {{4.0}}, {{4.1}}, {{4.2}}, {{4.3}}, {{4.4}}, {{4.5}}, {{4.6}}, {{4.7}}, {{4.8}}, {{4.9}}, {{5.0}}, {{5.1}}, {{5.2}}, {{5.3}}, {{5.4}}, {{5.5}}, {{5.6}}, {{5.7}}, {{5.8}}, {{5.9}}, {{6.0}}, {{6.1}}, {{6.2}}, {{6.3}}, {{6.4}}, {{6.5}}, {{6.6}}, {{6.7}}, {{6.8}}, {{6.9}}, {{7.0}}, {{7.1}}, {{7.2}}, {{7.3}}, {{7.4}}, {{7.5}}, {{7.6}}, {{7.7}}, {{7.8}}, {{7.9}}, {{8.0}}, {{8.1}}, {{8.2}}, {{8.3}}, {{8.4}}, {{8.5}}, {{8.6}}, {{8.7}}, {{8.8}}, {{8.9}}, {{9.0}}, {{9.1}}, {{9.2}}, {{9.3}}, {{9.4}}, {{9.5}}, {{9.6}}, {{9.7}}, {{9.8}}, {{9.9}}, {{10.0}}}};
	std::vector<std::vector<long double>> trainingOutputs = {{{{std::sin(0)}}, {{std::sin(0.1)}}, {{std::sin(0.2)}}, {{std::sin(0.3)}}, {{std::sin(0.4)}}, {{std::sin(0.5)}}, {{std::sin(0.6)}}, {{std::sin(0.7)}}, {{std::sin(0.8)}}, {{std::sin(0.9)}}, {{std::sin(1.0)}}, {{std::sin(1.1)}}, {{std::sin(1.2)}}, {{std::sin(1.3)}}, {{std::sin(1.4)}}, {{std::sin(1.5)}}, {{std::sin(1.6)}}, {{std::sin(1.7)}}, {{std::sin(1.8)}}, {{std::sin(1.9)}}, {{std::sin(2.0)}}, {{std::sin(2.1)}}, {{std::sin(2.2)}}, {{std::sin(2.3)}}, {{std::sin(2.4)}}, {{std::sin(2.5)}}, {{std::sin(2.6)}}, {{std::sin(2.7)}}, {{std::sin(2.8)}}, {{std::sin(2.9)}}, {{std::sin(3.0)}}, {{std::sin(3.1)}}, {{std::sin(3.2)}}, {{std::sin(3.3)}}, {{std::sin(3.4)}}, {{std::sin(3.5)}}, {{std::sin(3.6)}}, {{std::sin(3.7)}}, {{std::sin(3.8)}}, {{std::sin(3.9)}}, {{std::sin(4.0)}}, {{std::sin(4.1)}}, {{std::sin(4.2)}}, {{std::sin(4.3)}}, {{std::sin(4.4)}}, {{std::sin(4.5)}}, {{std::sin(4.6)}}, {{std::sin(4.7)}}, {{std::sin(4.8)}}, {{std::sin(4.9)}}, {{std::sin(5.0)}}, {{std::sin(5.1)}}, {{std::sin(5.2)}}, {{std::sin(5.3)}}, {{std::sin(5.4)}}, {{std::sin(5.5)}}, {{std::sin(5.6)}}, {{std::sin(5.7)}}, {{std::sin(5.8)}}, {{std::sin(5.9)}}, {{std::sin(6.0)}}, {{std::sin(6.1)}}, {{std::sin(6.2)}}, {{std::sin(6.3)}}, {{std::sin(6.4)}}, {{std::sin(6.5)}}, {{std::sin(6.6)}}, {{std::sin(6.7)}}, {{std::sin(6.8)}}, {{std::sin(6.9)}}, {{std::sin(7.0)}}, {{std::sin(7.1)}}, {{std::sin(7.2)}}, {{std::sin(7.3)}}, {{std::sin(7.4)}}, {{std::sin(7.5)}}, {{std::sin(7.6)}}, {{std::sin(7.7)}}, {{std::sin(7.8)}}, {{std::sin(7.9)}}, {{std::sin(8.0)}}, {{std::sin(8.1)}}, {{std::sin(8.2)}}, {{std::sin(8.3)}}, {{std::sin(8.4)}}, {{std::sin(8.5)}}, {{std::sin(8.6)}}, {{std::sin(8.7)}}, {{std::sin(8.8)}}, {{std::sin(8.9)}}, {{std::sin(9.0)}}, {{std::sin(9.1)}}, {{std::sin(9.2)}}, {{std::sin(9.3)}}, {{std::sin(9.4)}}, {{std::sin(9.5)}}, {{std::sin(9.6)}}, {{std::sin(9.7)}}, {{std::sin(9.8)}}, {{std::sin(9.9)}}, {{std::sin(10.0)}}}};
	std::shared_ptr<NeuralNetwork> neuralNetworkPointer(
		new NeuralNetwork(std::vector<unsigned long>({1, 14, 17, 23, 11, 13, 1}), NeuralNetwork::Tanh)
	);
	auto &network = *neuralNetworkPointer;
	Visualizer visualizer(network, 640, 480);
	auto trainingInputsSize = trainingInputs.size();
	network.learningRate = 0.0015;
	unsigned long trainingIteration = 0;
	auto trainingIterations = 100000;
	logger(Logger::Blank, "Training " + std::to_string(trainingIterations) + " iterations");
	for (; trainingIteration < trainingIterations; trainingIteration++)
	{
		for (unsigned long trainingIndex = 0; trainingIndex < trainingInputsSize; trainingIndex++)
		{
			auto &input = trainingInputs[trainingIndex];
			auto &output = trainingOutputs[trainingIndex];
			network.feedforward(input);
			network.backpropagate(output);
		}
		if (trainingIteration % 5000 == 0)
		{
			logger(Logger::Blank, "Trained " + std::to_string(trainingIteration) + " iterations");
		}
	}
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
			assert(difference <= tolerance);
		}
	}
	// 0.5
	std::vector<long double> input({0.5});
	network.feedforward(input);
	auto output = network.getOutputs();
	logger(Logger::Info, "y = sin(" + std::to_string(input[0]) + "). y = " + std::to_string(output[0]));
	// M_PI
	input[0] = M_PI;
	network.feedforward(input);
	output = network.getOutputs();
	logger(Logger::Info, "y = sin(" + std::to_string(input[0]) + "). y = " + std::to_string(output[0]));
	// M_PI / 2
	input[0] = M_PI /2;
	network.feedforward(input);
	output = network.getOutputs();
	logger(Logger::Info, "y = sin(" + std::to_string(input[0]) + "). y = " + std::to_string(output[0]));
	return 0;
};
/*
 */