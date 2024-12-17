/*
*/
#include <Layer.hpp>
using namespace zeuron;
/*
 */
Layer::Layer(const unsigned long &numberOfNeurons, const unsigned long &numberOfInputsPerNeuron, const ActivationType &activationType)
{
	for (unsigned long neuronIndex = 0; neuronIndex < numberOfNeurons; ++neuronIndex)
	{
		neurons.push_back({activationType, numberOfInputsPerNeuron});
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