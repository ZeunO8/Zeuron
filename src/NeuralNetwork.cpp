/*
 */
#include <NeuralNetwork.hpp>
#include <Logger.hpp>
#include <cmath>
#include <ByteStream.hpp>
using namespace zeuron;
using namespace bs;
/*
 */
NeuralNetwork::NeuralNetwork(const unsigned long &firstLayerSize,
														 const std::vector<std::pair<ActivationType, unsigned long>> &layerSpecs,
														 const long double &learningRate,
														 const long double &clipGradientValue):
	learningRate(learningRate),
	clipGradientValue(clipGradientValue)
{
	layers.push_back({firstLayerSize, 0, ActivationType::None});
	for (const auto &layerSpec : layerSpecs)
	{
		const ActivationType &activationType = layerSpec.first;
		unsigned long numberOfNeurons = layerSpec.second;
		unsigned long numberOfInputs = layers.back().neurons.size();
		layers.push_back({numberOfNeurons, numberOfInputs, activationType});
		activationTypes.push_back((int)activationType);
		activations.push_back(std::get<0>(activationDerivatives[activationType]));
		derivatives.push_back(std::get<1>(activationDerivatives[activationType]));
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
	if (!byteStream.read(activationTypes, bytesRead, true))
	{
		return;
	}
	for (auto &activationTypeInt : activationTypes)
	{
		auto activationType = (ActivationType)activationTypeInt;
		activations.push_back(std::get<0>(activationDerivatives[activationType]));
		derivatives.push_back(std::get<1>(activationDerivatives[activationType]));
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
		auto &activation = activations[layerIndex - 1];
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
const long double GRADIENT_CLIP_THRESHOLD = 10.0; // You can adjust this threshold

void NeuralNetwork::clipGradient(long double& gradient)
{
	if (clipGradientValue == -1.0)
		return;
	if (gradient > clipGradientValue)
		gradient = clipGradientValue;
	else if (gradient < -clipGradientValue)
		gradient = -clipGradientValue;
}
void NeuralNetwork::backpropagate(const std::vector<long double> &targetValues)
{
    Layer &outputLayer = layers.back();
    auto outputLayerNeuronsSize = outputLayer.neurons.size();
    auto outputLayerNeuronsData = outputLayer.neurons.data();
    auto targetValuesData = targetValues.data();
    auto &outputDerivative = derivatives.back();
    for (int i = 0; i < outputLayerNeuronsSize; ++i)
    {
        long double delta = targetValuesData[i] - outputLayerNeuronsData[i].outputValue;
        outputLayerNeuronsData[i].gradient = delta * outputDerivative(outputLayerNeuronsData[i].outputValue);
        clipGradient(outputLayerNeuronsData[i].gradient);
    }
    auto layersSize = layers.size();
    auto layersData = layers.data();
    for (int layerIndex = layersSize - 2; layerIndex > 0; --layerIndex)
    {
        Layer &hiddenLayer = layersData[layerIndex];
        Layer &nextLayer = layersData[layerIndex + 1];
        auto hiddenLayerNeuronsSize = hiddenLayer.neurons.size();
        auto hiddenLayerNeuronsData = hiddenLayer.neurons.data();
        auto nextLayerNeuronsSize = nextLayer.neurons.size();
        auto nextLayerNeuronsData = nextLayer.neurons.data();
        auto &layerDerivative = derivatives[layerIndex - 1];
        for (int neuronIndex = 0; neuronIndex < hiddenLayerNeuronsSize; ++neuronIndex)
        {
            long double error = 0.0;
            for (size_t nextNeuronIndex = 0; nextNeuronIndex < nextLayerNeuronsSize; ++nextNeuronIndex)
            {
                error += nextLayerNeuronsData[nextNeuronIndex].weights[neuronIndex] * nextLayerNeuronsData[nextNeuronIndex].gradient;
            }
            hiddenLayerNeuronsData[neuronIndex].gradient = error * layerDerivative(hiddenLayerNeuronsData[neuronIndex].outputValue);
            clipGradient(hiddenLayerNeuronsData[neuronIndex].gradient);
        }
    }
    for (int layerIndex = 1; layerIndex < layersSize; ++layerIndex)
    {
        Layer &layer = layersData[layerIndex];
        Layer &prevLayer = layersData[layerIndex - 1];
        auto prevLayerNeuronsData = prevLayer.neurons.data();
        for (Neuron &neuron : layer.neurons)
        {
            auto weightsSize = neuron.weights.size();
            auto neuronWeightsData = neuron.weights.data();
            for (int w = 0; w < weightsSize; ++w)
            {
                neuronWeightsData[w] += learningRate * neuron.gradient * prevLayerNeuronsData[w].outputValue;
            }
            neuron.bias += learningRate * neuron.gradient;
        }
    }
};
long double NeuralNetwork::calculateLoss(const std::vector<long double> &targetValues) const
{
	long double totalLoss = 0.0;
	const auto &outputLayer = layers.back();
	for (size_t i = 0; i < targetValues.size(); ++i)
	{
		auto delta = targetValues[i] - outputLayer.neurons[i].outputValue;
		totalLoss += delta * delta; // Mean Squared Error
	}
	return totalLoss / targetValues.size();
};
void NeuralNetwork::reward(const long double &rewardRate)
{
	for (auto &layer : layers)
	{
		for (auto &neuron : layer.neurons)
		{
			for (auto &weight : neuron.weights)
			{
				weight += rewardRate * weight;
			}
			neuron.bias += rewardRate * neuron.bias;
			clipGradient(neuron.bias); // Clip to avoid instability
		}
	}
};
void NeuralNetwork::penalize(const long double &penaltyRate)
{
	for (Layer &layer : layers)
	{
		for (Neuron &neuron : layer.neurons)
		{
			for (long double &weight : neuron.weights)
			{
				weight -= penaltyRate * weight;
			}
			neuron.bias -= penaltyRate * neuron.bias;
			clipGradient(neuron.bias); // Clip to avoid instability
		}
	}
};
/*
 */
const std::vector<long double> NeuralNetwork::getOutputs() const
{
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
	byteStream.write<const std::vector<int> &>(activationTypes);
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
const long double reluActivation(const long double &x)
{
	return (x > 0.0) ? x : 0.0; // ReLU: Returns x if x > 0, otherwise 0
}
const long double reluDerivative(const long double &x)
{
	return (x > 0.0) ? 1.0 : 0.0; // Derivative: 1 if x > 0, otherwise 0
}
/*
 */
const long double leakyReluActivation(const long double &x)
{
	long double result = (x > 0.0) ? x : 0.01 * x;
	return result;
}
const long double leakyReluDerivative(const long double &x)
{
	long double result = (x > 0.0) ? 1.0 : 0.01;
	return result;
}
/*
 */
const long double softplusActivation(const long double &x)
{
	return std::log(1.0 + exp(x));
};
const long double softplusDerivative(const long double &x)
{
	return 1.0 / (1.0 + exp(-x)); // Equivalent to sigmoid activation
};
/*
 */
const long double gaussianActivation(const long double &x)
{
	return exp(-x * x);
};
const long double gaussianDerivative(const long double &x)
{
	return -2.0 * x * exp(-x * x);
};
/*
 */
const long double softsignActivation(const long double &x)
{
	return x / (1.0 + std::abs(x));
};
const long double softsignDerivative(const long double &x)
{
	const long double denom = 1.0 + std::abs(x);
	return 1.0 / (denom * denom);
};
/*
 */
const long double bentIdentityActivation(const long double &x)
{
	return (std::sqrt(x * x + 1.0) - 1.0) / 2.0 + x;
};
const long double bentIdentityDerivative(const long double &x)
{
	return x / (2.0 * std::sqrt(x * x + 1.0)) + 1.0;
};
/*
 */
const long double arctanActivation(const long double &x)
{
	return std::atan(x);
};
const long double arctanDerivative(const long double &x)
{
	return 1.0 / (1.0 + x * x);
};
/*
 */
const long double sinusoidActivation(const long double &x)
{
	return std::sin(x);
};
const long double sinusoidDerivative(const long double &x)
{
	return std::cos(x);
};
/*
 */
const long double hardSigmoidActivation(const long double &x)
{
	return std::max(0.0L, std::min(1.0L, 0.2 * x + 0.5));
};
const long double hardSigmoidDerivative(const long double &x)
{
	return (x > -2.5 && x < 2.5) ? 0.2 : 0.0;
};
/*
 */
const long double mishActivation(const long double &x)
{
	return x * std::tanh(std::log(1.0 + exp(x)));
};
const long double mishDerivative(const long double &x)
{
	const long double sp = 1.0 / (1.0 + exp(-x)); // Sigmoid
	const long double omega = 4.0 * (x + 1.0) + 4.0 * exp(2.0 * x) + exp(3.0 * x) + exp(x) * (6.0 + 4.0 * x);
	const long double delta = 2.0 * (exp(x) + 1.0);
	return sp * omega / (delta * delta);
};
/*
 */
NeuralNetwork::ActivationDerivativesMap NeuralNetwork::activationDerivatives = {
	{ActivationType::Sigmoid, {sigmoidActivation, sigmoidDerivative}},
	{ActivationType::Tanh, {tanhActivation, tanhDerivative}},
	{ActivationType::Linear, {linearActivation, linearDerivative}},
	{ActivationType::Swish, {swishActivation, swishDerivative}},
	{ActivationType::ReLU, {reluActivation, reluDerivative}},
	{ActivationType::LeakyReLU, {leakyReluActivation, leakyReluDerivative}},
	{ActivationType::Softplus, {softplusActivation, softplusDerivative}},
	{ActivationType::Gaussian, {gaussianActivation, gaussianDerivative}},
	{ActivationType::Softsign, {softsignActivation, softsignDerivative}},
	{ActivationType::BentIdentity, {bentIdentityActivation, bentIdentityDerivative}},
	{ActivationType::Arctan, {arctanActivation, arctanDerivative}},
	{ActivationType::Sinusoid, {sinusoidActivation, sinusoidDerivative}},
	{ActivationType::HardSigmoid, {hardSigmoidActivation, hardSigmoidDerivative}}
};
/*
 */