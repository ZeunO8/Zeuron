/*
*/
#pragma once
#include "./Layer.hpp"
/*
 */
namespace nnpp
{
	struct NeuralNetwork
	{
		std::vector<Layer> layers;
		long double learningRate = 0.13;
		NeuralNetwork() = default;
		NeuralNetwork(const std::vector<unsigned long> &layerSizes);
		NeuralNetwork(const NeuralNetwork &) = delete;
		NeuralNetwork(NeuralNetwork &&) = delete;
		void print();
		void feedforward(const std::vector<long double> &inputValues);
		void backpropagate(const std::vector<long double> &targetValues);
		const std::vector<long double> getOutputs() const;
	};
}
/*
 */