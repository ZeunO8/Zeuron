/*
*/
#include <Neuron.hpp>
#include <Random.hpp>
using namespace nnpp;
/*
 */
Neuron::Neuron(const unsigned long &numberOfInputs,
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
	for (unsigned long weightIndex = weightsSize; weightIndex < numberOfInputs; weightIndex++)
	{
		this->weights.push_back(Random::value<long double>(-1, 1));
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