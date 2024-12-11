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
		Layer(const unsigned long &numberOfNeurons, const unsigned long &numberOfInputsPerNeuron);
		Layer &operator=(const Layer &other);
	};
}
/*
 */