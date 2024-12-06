/*
*/
#pragma once
#include <vector>
/*
 */
namespace nnpp
{
	struct Neuron
	{
		long double bias = 0;
		long double gradient = 0;
		std::vector<long double> weights;
		long double outputValue = 0;
		long double inputValue = 0;
		Neuron() = default;
		Neuron(const unsigned long &numberOfInputs,
					 const long double &bias = -1,
					 const long double &gradient = -1,
					 const std::vector<long double> &weights = {});
		Neuron &operator=(const Neuron &other);
	};
}
/*
 */