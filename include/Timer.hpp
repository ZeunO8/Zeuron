#pragma once
#include <chrono>

namespace zeuron
{
	struct Timer {
		using Clock = std::chrono::high_resolution_clock;
		using TimePoint = std::chrono::time_point<Clock>;

		// Member variables
		TimePoint start_time{};
		TimePoint stop_time{};
		bool running = false;
		double elapsed_seconds = 0.0;

		// Starts the timer
		void start();

		// Stops the timer and calculates elapsed time
		void stop();

		// Resets the timer
		void reset();

		// Accessor for elapsed time
		double getElapsedTime() const;
	};
}