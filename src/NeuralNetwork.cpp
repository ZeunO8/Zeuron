/*
 */
#include <NeuralNetwork.hpp>
#include <Logger.hpp>
#include <cmath>
#include <ByteStream.hpp>
using namespace nnpp;
using namespace bs;
/*
 */
NeuralNetwork::NeuralNetwork(const std::vector<unsigned long> &layerSizes, const ActivationType &activationType):
	activationType(activationType),
	activation(std::get<0>(activationDerivatives[activationType])),
	derivative(std::get<1>(activationDerivatives[activationType]))
{
	auto layerSizesSize = layerSizes.size();
	for (unsigned long layerIndex = 0; layerIndex < layerSizesSize; ++layerIndex)
	{
		unsigned long numberOfInputs = (layerIndex == 0) ? layerSizes[0] : layerSizes[layerIndex - 1];
		layers.push_back({layerSizes[layerIndex], numberOfInputs});
	}
};
/*
 */
template <>
const unsigned long ByteStream::write(const Neuron &neuron)
{
	unsigned long bytesWritten = 0;
	bytesWritten += write<const long double &>(neuron.bias);
	bytesWritten += write<const long double &>(neuron.gradient);
	bytesWritten += write<const std::vector<long double> &>(neuron.weights);
	bytesWritten += write<const long double &>(neuron.outputValue);
	bytesWritten += write<const long double &>(neuron.inputValue);
	return bytesWritten;
}
/*
 */
template <>
const bool ByteStream::read(Neuron &neuron, unsigned long &bytesRead, const bool &removeBytes)
{
	if (!read(neuron.bias, bytesRead, removeBytes))
	{
		return false;
	}
	if (!read(neuron.gradient, bytesRead, removeBytes))
	{
		return false;
	}
	if (!read(neuron.weights, bytesRead, removeBytes))
	{
		return false;
	}
	if (!read(neuron.outputValue, bytesRead, removeBytes))
	{
		return false;
	}
	if (!read(neuron.inputValue, bytesRead, removeBytes))
	{
		return false;
	}
	return true;
};
/*
 */
BYTE_STREAM_READ_VECTOR(Neuron);
BYTE_STREAM_WRITE_VECTOR(Neuron);

/*
 */
template <>
const unsigned long ByteStream::write(const Layer &layer)
{
	unsigned long bytesWritten = 0;
	bytesWritten += write<const std::vector<Neuron> &>(layer.neurons);
	return bytesWritten;
}
/*
 */
template <>
const bool ByteStream::read(Layer &layer, unsigned long &bytesRead, const bool &removeBytes)
{
	return read(layer.neurons, bytesRead, removeBytes);
};
/*
 */
BYTE_STREAM_READ_VECTOR(Layer);
BYTE_STREAM_WRITE_VECTOR(Layer);
/*
 */
NeuralNetwork::NeuralNetwork(bs::ByteStream& byteStream)
{
	unsigned long bytesRead = 0;
	if (!byteStream.read(learningRate, bytesRead, true))
	{
		return;
	}
	if (!byteStream.read(layers, bytesRead, true))
	{
		return;
	}
};
/*
 */
void NeuralNetwork::print()
{
	auto layersSize = layers.size();
	for (unsigned long layerIndex = 0; layerIndex < layersSize; layerIndex++)
	{
		logger(Logger::Blank, "Layer: " + std::to_string(layerIndex));
		auto &layer = layers[layerIndex];
		auto neuronsSize = layer.neurons.size();
		for (unsigned long neuronIndex = 0; neuronIndex < neuronsSize; neuronIndex++)
		{
			auto &neuron = layer.neurons[neuronIndex];
			logger(Logger::Blank,
				"\tNeuron: " + std::to_string(neuronIndex) +
					", inputValue: " + std::to_string(neuron.inputValue) +
					", outputValue: " +  std::to_string(neuron.outputValue) +
					", bias: " +  std::to_string(neuron.bias) +
					", gradient: " + std::to_string(neuron.gradient)
			);
		}
	}
}
/*
 */
void NeuralNetwork::feedforward(const std::vector<long double> &inputValues)
{
	std::lock_guard<std::mutex> lock(mutex);
	// Assign input values to the first layer
	auto layersSize = layers.size();
	auto layersData = layers.data();
	auto layer0NeuronsSize = layersData[0].neurons.size();
	auto layer0NeuronsData = layersData[0].neurons.data();
	for (size_t i = 0; i < layer0NeuronsSize; ++i)
	{
		layer0NeuronsData[i].outputValue = inputValues[i];
	}
	// Forward propagate through subsequent layers
	for (size_t layerIndex = 1; layerIndex < layersSize; ++layerIndex)
	{
		auto &prevLayer = layers[layerIndex - 1];
		auto prevLayerNeuronsSize = prevLayer.neurons.size();
		auto prevLayerNeuronsData = prevLayer.neurons.data();
		for (auto &neuron : layersData[layerIndex].neurons)
		{
			neuron.inputValue = 0.0; // Reset the input value
			auto neuronWeightsData = neuron.weights.data();
			for (unsigned long n = 0; n < prevLayerNeuronsSize; ++n)
			{
				// Accumulate the weighted input values
				neuron.inputValue += prevLayerNeuronsData[n].outputValue * neuronWeightsData[n];
			}
			// Add the bias and apply the activation function
			neuron.inputValue += neuron.bias;
			neuron.outputValue = activation(neuron.inputValue);
		}
	}
};
/*
 */
void NeuralNetwork::backpropagate(const std::vector<long double> &targetValues)
{
	std::lock_guard<std::mutex> lock(mutex);
	// Calculate gradients for the output layer
	Layer &outputLayer = layers.back();
	auto outputLayerNeuronsSize = outputLayer.neurons.size();
	auto outputLayerNeuronsData = outputLayer.neurons.data();
	auto targetValuesData = targetValues.data();
	for (size_t i = 0; i < outputLayerNeuronsSize; ++i)
	{
		long double delta = targetValuesData[i] - outputLayerNeuronsData[i].outputValue;
		outputLayerNeuronsData[i].gradient = delta * derivative(outputLayerNeuronsData[i].outputValue);
	}

	// Calculate gradients for the hidden layers (in reverse order)
	auto layersSize = layers.size();
	auto layersData = layers.data();
	for (size_t layerIndex = layersSize - 2; layerIndex > 0; --layerIndex)
	{
		Layer &hiddenLayer = layersData[layerIndex];
		Layer &nextLayer = layersData[layerIndex + 1];
		auto hiddenLayerNeuronsSize = hiddenLayer.neurons.size();
		auto hiddenLayerNeuronsData = hiddenLayer.neurons.data();
		auto nextLayerNeuronsSize = nextLayer.neurons.size();
		auto nextLayerNeuronsData = nextLayer.neurons.data();
		for (size_t neuronIndex = 0; neuronIndex < hiddenLayerNeuronsSize; ++neuronIndex)
		{
			long double error = 0.0;
			for (size_t nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronsSize; ++nextNeuronIndex)
			{
				error += nextLayerNeuronsData[nextNeuronIndex].weights[neuronIndex] * nextLayerNeuronsData[nextNeuronIndex].gradient;
			}
			hiddenLayerNeuronsData[neuronIndex].gradient = error * derivative(hiddenLayerNeuronsData[neuronIndex].outputValue);
		}
	}

	// Update weights and biases for all layers (except input layer)
	for (size_t layerIndex = 1; layerIndex < layersSize; ++layerIndex)
	{
		Layer &layer = layersData[layerIndex];
		Layer &prevLayer = layersData[layerIndex - 1];
		auto prevLayerNeuronsData = prevLayer.neurons.data();
		for (Neuron &neuron : layer.neurons)
		{
			auto weightsSize = neuron.weights.size();
			auto neuronWeightsData = neuron.weights.data();
			for (size_t w = 0; w < weightsSize; ++w)
			{
				neuronWeightsData[w] += learningRate * neuron.gradient * prevLayerNeuronsData[w].outputValue;
			}
			neuron.bias += learningRate * neuron.gradient;
		}
	}
};
/*
 */
const std::vector<long double> NeuralNetwork::getOutputs() const
{
	std::lock_guard<std::mutex> lock((std::mutex&)mutex);
	auto &lastLayer = layers.back();
	std::vector<long double> outputs;
	auto neuronsSize = lastLayer.neurons.size();
	auto neuronsData = lastLayer.neurons.data();
	for (unsigned long neuronIndex = 0; neuronIndex < neuronsSize; neuronIndex++)
	{
		outputs.push_back(neuronsData[neuronIndex].outputValue);
	}
	return outputs;
};
/*
 */
ByteStream NeuralNetwork::serialize() const
{
	ByteStream byteStream;
	byteStream.write<const long double &>(learningRate);
	byteStream.write<const std::vector<Layer> &>(layers);
	return byteStream;
};
/*
 */
const long double sigmoidActivation(const long double &x)
{
	return 1.0 / (1.0 + exp(-x));
};
const long double sigmoidDerivative(const long double &x)
{
	return x * (1.0 - x);
};
/*
 */
const long double tanhActivation(const long double &x)
{
	return std::tanh(x); // Maps x to [-1, 1]
};
const long double tanhDerivative(const long double &x)
{
	const long double tanhX = std::tanh(x);
	return 1.0 - tanhX * tanhX; // Derivative of tanh
};
/*
 */
const long double linearActivation(const long double &x)
{
	return x; // Identity function
};
const long double linearDerivative(const long double &x)
{
	return 1.0; // Constant derivative
};
/*
 */
const long double swishActivation(const long double &x)
{
	return x / (1.0 + exp(-x));
};
const long double swishDerivative(const long double &x)
{
	const long double sigmoidX = 1.0 / (1.0 + exp(-x));
	return sigmoidX + x * sigmoidX * (1.0 - sigmoidX); // Swish derivative
};
/*
 */
NeuralNetwork::ActivationDerivativesMap NeuralNetwork::activationDerivatives = {
	{NeuralNetwork::Sigmoid, {sigmoidActivation, sigmoidDerivative}},
{NeuralNetwork::Tanh, {tanhActivation, tanhDerivative}},
{NeuralNetwork::Linear, {linearActivation, linearDerivative}},
{NeuralNetwork::Swish, {swishActivation, swishDerivative}}
};
/*
 */