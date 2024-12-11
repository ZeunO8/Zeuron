#include <Timer.hpp>
#include <Logger.hpp>
using namespace nnpp;

// Starts the timer
void Timer::start() {
	start_time = Clock::now();
	running = true;
}

// Stops the timer and calculates elapsed time
void Timer::stop() {
	if (running) {
		stop_time = Clock::now();
		elapsed_seconds = std::chrono::duration<double>(stop_time - start_time).count();
		running = false;
	} else {
		logger(Logger::Info, "Timer is not running. Call start() first.");
	}
}

// Resets the timer
void Timer::reset() {
	start_time = TimePoint{};
	stop_time = TimePoint{};
	running = false;
	elapsed_seconds = 0.0;
}

// Accessor for elapsed time
double Timer::getElapsedTime() const {
	if (running) {
		logger(Logger::Info, "Timer is still running. Stop it to get elapsed time.");
		return 0.0;
	}
	return elapsed_seconds;
}