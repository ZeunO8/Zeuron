/*
 */
#include <Visualizer.hpp>
#include <fenster.h>
#include <bit>
#include <numeric>
#include <algorithm>
using namespace nnpp;
/*
 */
Visualizer::Visualizer(NeuralNetwork& network, const int &windowWidth, const int &windowHeight):
	network(network),
	windowThread(&Visualizer::startWindow, this),
	windowWidth(windowWidth),
	windowHeight(windowHeight),
	buf((uint32_t*)malloc(windowWidth * windowHeight * sizeof(uint32_t)), free),
	f(new struct fenster({ "nnpp visualizer", windowWidth, windowHeight, buf.get()}))
{
};
/*
 */
Visualizer::~Visualizer()
{
	fenster_close(f);
	windowThread.join();
	delete f;
};
/*
 */
// Helper function to map a value to a color (gradient) in RGBA format
Color mapValueToColor(long double value, long double minVal, long double maxVal)
{
	// Clamp value between minVal and maxVal
	value = (std::max)(minVal, (std::min)(value, maxVal));

	// Normalize value to [0, 1]
	float normalized = (value - minVal) / (maxVal - minVal);

	// Map to RGB: Green for positive, Red for negative
	uint8_t red = static_cast<uint8_t>((1.0f - normalized) * 255);   // Red decreases as value increases
	uint8_t green = static_cast<uint8_t>(normalized * 255);         // Green increases as value increases
	uint8_t blue = 0;                                               // Blue is not used here

	return {blue, green, red, 255}; // Return RGBA color
}

// Helper function to convert Color to uint32_t
uint32_t colorToUint32(const Color &color)
{
	return std::bit_cast<uint32_t>(color);
}
/*
 */
void Visualizer::render()
{
    static const int radius = 10;
    static const uint32_t defaultLineColor = 0x00333333; // Dark grey color for the lines

    int numLayers = network.layers.size();

    // Horizontal spacing between layers
    int layerSpacing = (windowWidth - 2 * radius) / (numLayers > 1 ? numLayers - 1 : 1);
    int centerX = windowWidth / 2;
    int centerY = windowHeight / 2;

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
        int neuronSpacing = (windowHeight - 2 * radius) / (numNeurons > 1 ? numNeurons - 1 : 1);

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

                    fenster_line(f, prevX, prevY, nextX, nextY, lineColor);
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
        int neuronSpacing = (windowHeight - 2 * radius) / (numNeurons > 1 ? numNeurons - 1 : 1);

        // Initial y-coordinate (start from the center of the neurons in the layer)
        int y = centerY - (numNeurons - 1) * neuronSpacing / 2;

        // Draw the neurons (circles)
        for (auto &neuron : layer.neurons)
        {
            // Color the neuron based on its weights (average weight)
            uint32_t neuronColor = mapWeightToColor(neuron);

            // Draw the neuron circle
            fenster_circle(f, x, y, radius, neuronColor);
            y += neuronSpacing;
        }

        x += layerSpacing;
    }
};

// Helper function to map a neuron output value to a color
uint32_t Visualizer::mapValueToColor(long double value)
{
	// Ensure value is between 0 and 1
	value = std::clamp(value, 0.0L, 1.0L);

	// Map value to a grayscale color (from black to white)
	uint8_t color = static_cast<uint8_t>(value * 255);
	return (color << 16) | (color << 8) | color;  // RGB format
}

// Helper function to map the weights to a color
uint32_t Visualizer::mapWeightToColor(const Neuron &neuron)
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
	return (r << 16) | (b << 8);  // RGB format (no green for simplicity)
}

void Visualizer::startWindow()
{
	fenster_open(f);
	uint32_t t = 0;
	int64_t now = fenster_time();
	while (fenster_loop(f) == 0)
	{
		render();
		int64_t time = fenster_time();
		if (time - now < 1000 / 60)
			fenster_sleep(time - now);
		now = time;
	}
};
/*
 */
