/*
 */
#pragma once
#include <NeuralNetwork.hpp>
#include <thread>
/*
 */
struct fenster;
namespace nnpp
{
	struct Color
	{
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	};
	struct Visualizer
  {
    NeuralNetwork &network;
		std::thread windowThread;
		unsigned int windowWidth;
		unsigned int windowHeight;
		std::shared_ptr<uint32_t> buf;
		struct fenster *f;
  	Visualizer(NeuralNetwork &network, const int &windowWidth, const int &windowHeight);
		void close();
		~Visualizer();
    void render();
		uint32_t mapValueToColor(long double value);
		uint32_t mapWeightToColor(const Neuron &neuron);
		void startWindow();
  };
}
/*
 */