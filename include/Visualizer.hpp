/*
 */
#pragma once
#include <NeuralNetwork.hpp>
#include <thread>
#include <memory>
#include <anex/modules/fenster/Fenster.hpp>
/*
 */
namespace zeuron
{
	using namespace anex::modules::fenster;
	struct Color
	{
		uint8_t b;
		uint8_t g;
		uint8_t r;
		uint8_t a;
	};
	struct VisualizerEntity : anex::IEntity
	{
		NeuralNetwork& network;
		VisualizerEntity(anex::IGame &game, NeuralNetwork& network);
		void render() override;
		uint32_t mapValueToColor(long double value);
		uint32_t mapWeightToColor(const Neuron &neuron);
	};
	struct VisualizerScene : anex::IScene
	{
		VisualizerScene(anex::IGame &game, NeuralNetwork& network);
	};
	struct Visualizer : FensterGame
  {
  	Visualizer(NeuralNetwork &network, const int &windowWidth, const int &windowHeight);
  };
}
/*
 */