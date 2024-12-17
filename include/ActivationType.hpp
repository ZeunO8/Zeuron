#pragma once

namespace zeuron
{
	enum class ActivationType
	{
		None = 0,
		Sigmoid,
		Linear,
		Tanh,
		Swish,
		ReLU,
		LeakyReLU,
		Softplus,
		Gaussian,
		Softsign,
		BentIdentity,
		Arctan,
		Sinusoid,
		HardSigmoid
	};
}