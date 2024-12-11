/*
*/
#pragma once
#include "./Layer.hpp"
#include <unordered_map>
#include <mutex>
/*
 */
namespace bs
{
	struct ByteStream;
}
namespace zeuron
{
	#define ActivationFunction const long double(*)(const long double &)
	#define DerivativeFunction const long double(*)(const long double &)
	struct NeuralNetwork
	{
		enum ActivationType
		{
			Sigmoid,
			Linear,
			Tanh,
			Swish,
			ReLU,
			LeakyReLU
		};
		typedef std::unordered_map<ActivationType, std::pair<ActivationFunction, DerivativeFunction>> ActivationDerivativesMap;
		static ActivationDerivativesMap activationDerivatives;
		std::vector<Layer> layers;
		long double learningRate;
		std::vector<int> activationTypes;
		std::vector<const long double(*)(const long double &)> activations;
		std::vector<const long double(*)(const long double &)> derivatives;
		std::mutex mutex;
		NeuralNetwork() = default;
		NeuralNetwork(const unsigned long &firstLayerSize,
									const std::vector<std::pair<ActivationType, unsigned long>> &layerSpecs,
									const long double &learningRate = 0.13);
		NeuralNetwork(bs::ByteStream &byteStream);
		NeuralNetwork(const NeuralNetwork &) = delete;
		NeuralNetwork(NeuralNetwork &&) = delete;
		void print();
		void feedforward(const std::vector<long double> &inputValues);
		void backpropagate(const std::vector<long double> &targetValues);
		const std::vector<long double> getOutputs() const;
		bs::ByteStream serialize() const;
	};
}
/*
 */