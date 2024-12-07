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
static void fenster_line(struct fenster *f, int x0, int y0, int x1, int y1,
                         uint32_t c) {
  int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = (dx > dy ? dx : -dy) / 2, e2;
  for (;;) {
    fenster_pixel(f, x0, y0) = c;
    if (x0 == x1 && y0 == y1) {
      break;
    }
    e2 = err;
    if (e2 > -dx) {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dy) {
      err += dx;
      y0 += sy;
    }
  }
}

static void fenster_rect(struct fenster *f, int x, int y, int w, int h,
                         uint32_t c) {
  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      fenster_pixel(f, x + col, y + row) = c;
    }
  }
}

static void fenster_circle(struct fenster *f, int x, int y, int r, uint32_t c) {
  for (int dy = -r; dy <= r; dy++) {
    for (int dx = -r; dx <= r; dx++) {
      if (dx * dx + dy * dy <= r * r) {
        fenster_pixel(f, x + dx, y + dy) = c;
      }
    }
  }
}

static void fenster_fill(struct fenster *f, int x, int y, uint32_t old,
                         uint32_t c) {
  if (x < 0 || y < 0 || x >= f->width || y >= f->height) {
    return;
  }
  if (fenster_pixel(f, x, y) == old) {
    fenster_pixel(f, x, y) = c;
    fenster_fill(f, x - 1, y, old, c);
    fenster_fill(f, x + 1, y, old, c);
    fenster_fill(f, x, y - 1, old, c);
    fenster_fill(f, x, y + 1, old, c);
  }
}

// clang-format off
static uint16_t font5x3[] = {0x0000,0x2092,0x002d,0x5f7d,0x279e,0x52a5,0x7ad6,0x0012,0x4494,0x1491,0x017a,0x05d0,0x1400,0x01c0,0x0400,0x12a4,0x2b6a,0x749a,0x752a,0x38a3,0x4f4a,0x38cf,0x3bce,0x12a7,0x3aae,0x49ae,0x0410,0x1410,0x4454,0x0e38,0x1511,0x10e3,0x73ee,0x5f7a,0x3beb,0x624e,0x3b6b,0x73cf,0x13cf,0x6b4e,0x5bed,0x7497,0x2b27,0x5add,0x7249,0x5b7d,0x5b6b,0x3b6e,0x12eb,0x4f6b,0x5aeb,0x388e,0x2497,0x6b6d,0x256d,0x5f6d,0x5aad,0x24ad,0x72a7,0x6496,0x4889,0x3493,0x002a,0xf000,0x0011,0x6b98,0x3b79,0x7270,0x7b74,0x6750,0x95d6,0xb9ee,0x5b59,0x6410,0xb482,0x56e8,0x6492,0x5be8,0x5b58,0x3b70,0x976a,0xcd6a,0x1370,0x38f0,0x64ba,0x3b68,0x2568,0x5f68,0x54a8,0xb9ad,0x73b8,0x64d6,0x2492,0x3593,0x03e0};
// clang-format on
static void fenster_text(struct fenster *f, int x, int y, char *s, int scale,
                         uint32_t c) {
  while (*s) {
    char chr = *s++;
    if (chr > 32) {
      uint16_t bmp = font5x3[chr - 32];
      for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 3; dx++) {
          if (bmp >> (dy * 3 + dx) & 1) {
            fenster_rect(f, x + dx * scale, y + dy * scale, scale, scale, c);
          }
        }
      }
    }
    x = x + 4 * scale;
  }
}
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
void Visualizer::close()
{
	fenster_close(f);
};
/*
 */
Visualizer::~Visualizer()
{
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
		fenster_rect(f, 0, 0, windowWidth, windowHeight, 0x0000bb99);
    static const int radius = 10;
    static const uint32_t defaultLineColor = 0x00555555; // Dark grey color for the lines

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
	return (r << 16) | (b << 0);  // RGB format (no green for simplicity)
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
