/*
*/
#include <NeuralNetwork.hpp>
#include <Logger.hpp>
#include <memory>
#include <cassert>
#define _USE_MATH_DEFINES
#include <math.h>
#include <Visualizer.hpp>
#include <fstream>
#include <iostream>
#include <ByteStream.hpp>
#include <Timer.hpp>
using namespace zeuron;
using namespace bs;
/*
 * Sinusoidal Function
 * Predict y=sin(x) for x values in the range [0,π].
 */
std::pair<std::shared_ptr<char>, unsigned long> readFileToBuffer(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open())
	{
		throw std::ios_base::failure("Error: Unable to open file for reading.");
	}
	std::streampos fileSize = file.tellg();
	if (fileSize <= 0)
	{
		throw std::ios_base::failure("Error: File is empty or has invalid size.");
	}
	unsigned long size = static_cast<unsigned long>(fileSize);
	std::shared_ptr<char> buffer(new char[size], std::default_delete<char[]>());
	file.seekg(0, std::ios::beg);
	file.read(buffer.get(), size);
	if (!file)
	{
		throw std::ios_base::failure("Error: Reading the file failed.");
	}
	file.close();
	return std::make_pair(buffer, size);
};
/*
 */
void writeBufferToFile(const char* buffer, unsigned long size, const std::string& filename)
{
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open())
	{
		std::cerr << "Error: Unable to open file for writing.\n";
		return;
	}
	file.write(buffer, static_cast<std::streamsize>(size));
	if (!file)
	{
		std::cerr << "Error: Writing to the file failed.\n";
	}
	file.close();
}
/*
 */
long double learningRateSchedule(const long double &initialLearningRate, const long double &decayRate, const unsigned long &iteration)
{
	return initialLearningRate * std::exp(-decayRate * iteration);
}
int main(int argc, char **argv)
{
	Timer timer;
	std::vector<std::vector<long double>> trainingInputs = {{{{0}}, {{0.1}}, {{0.2}}, {{0.3}}, {{0.4}}, {{0.5}}, {{0.6}}, {{0.7}}, {{0.8}}, {{0.9}}, {{1.0}}, {{1.1}}, {{1.2}}, {{1.3}}, {{1.4}}, {{1.5}}, {{1.6}}, {{1.7}}, {{1.8}}, {{1.9}}, {{2.0}}, {{2.1}}, {{2.2}}, {{2.3}}, {{2.4}}, {{2.5}}, {{2.6}}, {{2.7}}, {{2.8}}, {{2.9}}, {{3.0}}, {{3.1}}, {{3.2}}, {{3.3}}, {{3.4}}, {{3.5}}, {{3.6}}, {{3.7}}, {{3.8}}, {{3.9}}, {{4.0}}, {{4.1}}, {{4.2}}, {{4.3}}, {{4.4}}, {{4.5}}, {{4.6}}, {{4.7}}, {{4.8}}, {{4.9}}, {{5.0}}, {{5.1}}, {{5.2}}, {{5.3}}, {{5.4}}, {{5.5}}, {{5.6}}, {{5.7}}, {{5.8}}, {{5.9}}, {{6.0}}, {{6.1}}, {{6.2}}, {{6.3}}, {{6.4}}, {{6.5}}, {{6.6}}, {{6.7}}, {{6.8}}, {{6.9}}, {{7.0}}, {{7.1}}, {{7.2}}, {{7.3}}, {{7.4}}, {{7.5}}, {{7.6}}, {{7.7}}, {{7.8}}, {{7.9}}, {{8.0}}, {{8.1}}, {{8.2}}, {{8.3}}, {{8.4}}, {{8.5}}, {{8.6}}, {{8.7}}, {{8.8}}, {{8.9}}, {{9.0}}, {{9.1}}, {{9.2}}, {{9.3}}, {{9.4}}, {{9.5}}, {{9.6}}, {{9.7}}, {{9.8}}, {{9.9}}, {{10.0}}}};
	std::vector<std::vector<long double>> trainingOutputs = {{{{std::sin(0)}}, {{std::sin(0.1)}}, {{std::sin(0.2)}}, {{std::sin(0.3)}}, {{std::sin(0.4)}}, {{std::sin(0.5)}}, {{std::sin(0.6)}}, {{std::sin(0.7)}}, {{std::sin(0.8)}}, {{std::sin(0.9)}}, {{std::sin(1.0)}}, {{std::sin(1.1)}}, {{std::sin(1.2)}}, {{std::sin(1.3)}}, {{std::sin(1.4)}}, {{std::sin(1.5)}}, {{std::sin(1.6)}}, {{std::sin(1.7)}}, {{std::sin(1.8)}}, {{std::sin(1.9)}}, {{std::sin(2.0)}}, {{std::sin(2.1)}}, {{std::sin(2.2)}}, {{std::sin(2.3)}}, {{std::sin(2.4)}}, {{std::sin(2.5)}}, {{std::sin(2.6)}}, {{std::sin(2.7)}}, {{std::sin(2.8)}}, {{std::sin(2.9)}}, {{std::sin(3.0)}}, {{std::sin(3.1)}}, {{std::sin(3.2)}}, {{std::sin(3.3)}}, {{std::sin(3.4)}}, {{std::sin(3.5)}}, {{std::sin(3.6)}}, {{std::sin(3.7)}}, {{std::sin(3.8)}}, {{std::sin(3.9)}}, {{std::sin(4.0)}}, {{std::sin(4.1)}}, {{std::sin(4.2)}}, {{std::sin(4.3)}}, {{std::sin(4.4)}}, {{std::sin(4.5)}}, {{std::sin(4.6)}}, {{std::sin(4.7)}}, {{std::sin(4.8)}}, {{std::sin(4.9)}}, {{std::sin(5.0)}}, {{std::sin(5.1)}}, {{std::sin(5.2)}}, {{std::sin(5.3)}}, {{std::sin(5.4)}}, {{std::sin(5.5)}}, {{std::sin(5.6)}}, {{std::sin(5.7)}}, {{std::sin(5.8)}}, {{std::sin(5.9)}}, {{std::sin(6.0)}}, {{std::sin(6.1)}}, {{std::sin(6.2)}}, {{std::sin(6.3)}}, {{std::sin(6.4)}}, {{std::sin(6.5)}}, {{std::sin(6.6)}}, {{std::sin(6.7)}}, {{std::sin(6.8)}}, {{std::sin(6.9)}}, {{std::sin(7.0)}}, {{std::sin(7.1)}}, {{std::sin(7.2)}}, {{std::sin(7.3)}}, {{std::sin(7.4)}}, {{std::sin(7.5)}}, {{std::sin(7.6)}}, {{std::sin(7.7)}}, {{std::sin(7.8)}}, {{std::sin(7.9)}}, {{std::sin(8.0)}}, {{std::sin(8.1)}}, {{std::sin(8.2)}}, {{std::sin(8.3)}}, {{std::sin(8.4)}}, {{std::sin(8.5)}}, {{std::sin(8.6)}}, {{std::sin(8.7)}}, {{std::sin(8.8)}}, {{std::sin(8.9)}}, {{std::sin(9.0)}}, {{std::sin(9.1)}}, {{std::sin(9.2)}}, {{std::sin(9.3)}}, {{std::sin(9.4)}}, {{std::sin(9.5)}}, {{std::sin(9.6)}}, {{std::sin(9.7)}}, {{std::sin(9.8)}}, {{std::sin(9.9)}}, {{std::sin(10.0)}}}};
	std::shared_ptr<NeuralNetwork> neuralNetworkPointer;
	bool trained = false;
	try
	{
		auto bytesSizePair = readFileToBuffer("sinusoidal.nrl");
		ByteStream byteStream(std::get<1>(bytesSizePair), std::get<0>(bytesSizePair));
		neuralNetworkPointer = std::make_shared<NeuralNetwork>(byteStream);
		trained = argc == 1;
	}
	catch (...)
	{
		neuralNetworkPointer = std::make_shared<NeuralNetwork>(
			1,
			std::vector<std::pair<NeuralNetwork::ActivationType, unsigned long>>({
	        { NeuralNetwork::Tanh, 18 },  // Start with a moderate number of neurons
					{ NeuralNetwork::Tanh, 14 },   // Reduce the size progressively
					{ NeuralNetwork::LeakyReLU, 10 },   // Further reduce to improve learning efficiency
					{ NeuralNetwork::Tanh, 6 },        // Tanh here to allow non-linearity
					{ NeuralNetwork::Tanh, 1 }         // Output layer
			}),
			0.015
		);
	}
	auto &network = *neuralNetworkPointer;
	Visualizer visualizer(network, 640, 480);
	auto trainingInputsSize = trainingInputs.size();
	if (!trained)
	{
		timer.start();
		const auto initialLearningRate = network.learningRate;
		unsigned long trainingIteration = 0;
		auto trainingIterations = 150000;
		logger(Logger::Blank, "Training " + std::to_string(trainingIterations) + " iterations");
		for (; trainingIteration < trainingIterations; trainingIteration++)
		{
			network.learningRate = learningRateSchedule(initialLearningRate,  0.0002, trainingIteration);
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
		timer.stop();
		logger(Logger::Blank, "Trained " + std::to_string(trainingIteration) + " iterations in " + std::to_string(timer.getElapsedTime()) + " seconds");
	}
	static const long double tolerance = 0.05;
	timer.reset();
	timer.start();
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
						" } the network has a difference of: " + std::to_string(difference) +
						", output: " + std::to_string(actualOutputs[outputIndex]) +
						", is " + (difference <= tolerance ? "within" : "not within") + " tolerance of " + std::to_string(tolerance));
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
	timer.stop();
	logger(Logger::Info, "Tested NeuralNetwork in " + std::to_string(timer.getElapsedTime()) + " seconds");
	std::this_thread::sleep_for(std::chrono::seconds(5));
	visualizer.close();
	auto nnStream = network.serialize();
	writeBufferToFile(nnStream.bytes.get(), nnStream.bytesSize, "sinusoidal.nrl");
	return 0;
};
/*
 */