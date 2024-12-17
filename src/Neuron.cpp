/*
*/
#include <Neuron.hpp>
#include <Random.hpp>
using namespace zeuron;
/*
 */
Neuron::Neuron(const ActivationType &activationType,
							 const unsigned long &numberOfInputs,
							 const long double &bias,
							 const long double &gradient,
							 const std::vector<long double> &weights)
{
	if (bias == -1)
	{
		this->bias = 1;
	}
	else
	{
		this->bias = bias;
	}
	if (gradient != -1)
	{
		this->gradient = gradient;
	}
	this->weights = weights;
	auto weightsSize = this->weights.size();
	float stddev = getWeightStdDev(activationType, numberOfInputs);
	for (unsigned long weightIndex = weightsSize; weightIndex < numberOfInputs; weightIndex++)
	{
		this->weights.push_back(Random::value<long double>(-stddev, stddev));
	}
};
/*
 */
Neuron &Neuron::operator=(const Neuron &other)
{
	bias = other.bias;
	gradient = other.gradient;
	weights = other.weights;
	return *this;
};
/*
 */
long double Neuron::getWeightStdDev(const ActivationType &activationType, const unsigned long &numberOfInputs)
{
	switch (activationType)
	{
	case ActivationType::ReLU:
	case ActivationType::LeakyReLU:
	case ActivationType::Swish:
	case ActivationType::Softplus:
			return std::sqrt(2.0 / numberOfInputs); // He Initialization

	case ActivationType::Tanh:
	case ActivationType::Sigmoid:
	case ActivationType::Linear:
			return std::sqrt(1.0 / numberOfInputs); // Xavier Initialization

	case ActivationType::Softsign:
	case ActivationType::BentIdentity:
	case ActivationType::HardSigmoid:
			return std::sqrt(1.0 / numberOfInputs); // LeCun Initialization

	case ActivationType::Gaussian:
	case ActivationType::Sinusoid:
	case ActivationType::Arctan:
			return 0.01; // Small Random Initialization

	default:
		return 1.0;
	}
}
/*
 */