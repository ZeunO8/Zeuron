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
namespace nnpp
{
	#define ActivationFunction const long double(*)(const long double &)
	#define DerivativeFunction const long double(*)(const long double &)
	#define ActivationFunctionD(NAME) const long double(*NAME)(const long double &)
	#define DerivativeFunctionD(NAME) const long double(*NAME)(const long double &)
	struct NeuralNetwork
	{
		enum ActivationType
		{
			Sigmoid,
			Linear,
			Tanh,
			Swish
		};
		typedef std::unordered_map<ActivationType, std::pair<ActivationFunction, DerivativeFunction>> ActivationDerivativesMap;
		static ActivationDerivativesMap activationDerivatives;
		std::vector<Layer> layers;
		long double learningRate = 0.13;
		ActivationType activationType = Sigmoid;
		ActivationFunctionD(activation);
		DerivativeFunctionD(derivative);
		std::mutex mutex;
		NeuralNetwork() = default;
		NeuralNetwork(const std::vector<unsigned long> &layerSizes, const ActivationType &activationType = Sigmoid);
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