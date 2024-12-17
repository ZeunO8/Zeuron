/*
*/
#pragma once
#include "./Neuron.hpp"
/*
 */
namespace zeuron
{
	struct Layer
	{
		std::vector<Neuron> neurons;
		Layer() = default;
		Layer(const unsigned long &numberOfNeurons, const unsigned long &numberOfInputsPerNeuron, const ActivationType &activationType);
		Layer &operator=(const Layer &other);
	};
}
/*
 */