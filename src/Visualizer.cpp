/*
 */
#include <Visualizer.hpp>
#include <bit>
#include <numeric>
#include <algorithm>
using namespace zeuron;
VisualizerEntity::VisualizerEntity(anex::IGame &game, NeuralNetwork& network):
	IEntity(game),
    network(network)
{};
/*
 */
// Helper function to convert Color to uint32_t
uint32_t colorToUint32(const Color &color)
{
    return std::bit_cast<uint32_t>(color);
}
// Helper function to map a neuron output value to a color
uint32_t VisualizerEntity::mapValueToColor(long double value)
{
    // Ensure value is between 0 and 1
    value = std::clamp(value, 0.0L, 1.0L);

    // Map value to a grayscale color (from black to white)
    uint8_t color = static_cast<uint8_t>(value * 255);
    return (color << 16) | (color << 8) | color;  // RGB format
}

// Helper function to map the weights to a color
uint32_t VisualizerEntity::mapWeightToColor(const Neuron &neuron)
{
    long double avgWeight = 0.0;
    if (!neuron.weights.empty())
    {
        avgWeight = std::accumulate(neuron.weights.begin(), neuron.weights.end(), 0.0) / neuron.weights.size();
    }

    // Normalize weight value (range [-1, 1] to [0, 1])
    avgWeight = std::clamp((avgWeight + 1.0L) / 2.0L, 0.0L, 1.0L);

    // Map weight value to color (e.g., blue to red spectrum)
    uint8_t r = static_cast<uint8_t>(avgWeight * 255);
    uint8_t b = static_cast<uint8_t>((1.0 - avgWeight) * 255);
    return (r << 16) | (b << 0);  // RGB format (no green for simplicity)
}
void VisualizerEntity::render()
{
    FensterGame &fensterGame = (FensterGame &) game;
		fenster_rect(fensterGame.f, 0, 0, game.windowWidth, game.windowHeight, 0x0000bb99);
    static const int radius = 10;
    static const uint32_t defaultLineColor = 0x00555555; // Dark grey color for the lines

    int numLayers = network.layers.size();

    // Horizontal spacing between layers
    int layerSpacing = (game.windowWidth - 2 * radius) / (numLayers > 1 ? numLayers - 1 : 1);
    int centerX = game.windowWidth / 2;
    int centerY = game.windowHeight / 2;

    // Coordinates for the current and next layer neurons
    std::vector<std::pair<int, int>> currentLayerPositions;
    std::vector<std::pair<int, int>> nextLayerPositions;

    // Initial x-coordinate (start from the center of the leftmost layer)
    int x = centerX - (numLayers - 1) * layerSpacing / 2;

    // Pass 1: Draw all the lines between neurons
    for (size_t i = 0; i < numLayers; ++i)
    {
        auto &layer = network.layers[i];
        int numNeurons = layer.neurons.size();

        // Vertical spacing between neurons in the layer
        int neuronSpacing = (game.windowHeight - 2 * radius) / (numNeurons > 1 ? numNeurons - 1 : 1);

        // Initial y-coordinate (start from the center of the neurons in the layer)
        int y = centerY - (numNeurons - 1) * neuronSpacing / 2;

        nextLayerPositions.clear();

        // Save the positions and indices for the current layer
        std::vector<std::pair<int, int>> currentLayerNeurons;
        for (int j = 0; j < numNeurons; ++j)
        {
            currentLayerNeurons.emplace_back(x, y);
            nextLayerPositions.emplace_back(x, y);
            y += neuronSpacing;
        }

        // Draw connections to the next layer if it exists
        if (i > 0)
        {
            for (size_t prevNeuronIndex = 0; prevNeuronIndex < currentLayerPositions.size(); ++prevNeuronIndex)
            {
                auto &[prevX, prevY] = currentLayerPositions[prevNeuronIndex];

                for (size_t nextNeuronIndex = 0; nextNeuronIndex < nextLayerPositions.size(); ++nextNeuronIndex)
                {
                    auto &[nextX, nextY] = nextLayerPositions[nextNeuronIndex];

                    // Get the output value of the neuron in the previous layer
                    auto &prevLayerNeuron = network.layers[i - 1].neurons[prevNeuronIndex];
                    uint32_t lineColor = mapValueToColor(prevLayerNeuron.outputValue);

                    fenster_line(fensterGame.f, prevX, prevY, nextX, nextY, lineColor);
                }
            }
        }

        // Prepare for the next iteration
        currentLayerPositions = std::move(nextLayerPositions);
        x += layerSpacing;
    }

    // Pass 2: Draw all the neurons (circles)
    x = centerX - (numLayers - 1) * layerSpacing / 2;

    for (size_t i = 0; i < numLayers; ++i)
    {
        auto &layer = network.layers[i];
        int numNeurons = layer.neurons.size();

        // Vertical spacing between neurons in the layer
        int neuronSpacing = (game.windowHeight - 2 * radius) / (numNeurons > 1 ? numNeurons - 1 : 1);

        // Initial y-coordinate (start from the center of the neurons in the layer)
        int y = centerY - (numNeurons - 1) * neuronSpacing / 2;

        // Draw the neurons (circles)
        for (auto &neuron : layer.neurons)
        {
            // Color the neuron based on its weights (average weight)
            uint32_t neuronColor = mapWeightToColor(neuron);

            // Draw the neuron circle
            fenster_circle(fensterGame.f, x, y, radius, neuronColor);
            y += neuronSpacing;
        }

        x += layerSpacing;
    }

};
VisualizerScene::VisualizerScene(anex::IGame &game, NeuralNetwork& network):
	IScene(game)
{
	addEntity(std::make_shared<VisualizerEntity>(game, network));
};
/*
 */
Visualizer::Visualizer(NeuralNetwork& network, const int &windowWidth, const int &windowHeight):
	FensterGame(windowWidth, windowHeight)
{
    setIScene(std::make_shared<VisualizerScene>(*this, network));
};
/*
 */
