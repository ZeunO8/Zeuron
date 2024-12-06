/*
*/
#include <Layer.hpp>
using namespace nnpp;
/*
 */
Layer::Layer(const unsigned long &numberOfNeurons, const unsigned long &numberOfInputsPerNeuron)
{
	for (unsigned long neuronIndex = 0; neuronIndex < numberOfNeurons; ++neuronIndex)
	{
		neurons.push_back({numberOfInputsPerNeuron});
	}
};
/*
 */
Layer &Layer::operator=(const Layer &other)
{
	neurons = other.neurons;
	return *this;
};
/*
 */