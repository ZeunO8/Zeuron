
#include "Logger.hpp"
#include <iostream>

std::unordered_map<Logger::LogType, std::string> Logger::logTypeMap({
	{ Logger::Blank, "" },
	{ Logger::Info, "Info: " },
	{ Logger::Error, "Error: " }
});

void Logger::operator()(const Logger::LogType &logType, const std::string& str)
{
	auto logTypeIter = logTypeMap.find(logType);
	if (logTypeIter == logTypeMap.end())
	{
		throw std::runtime_error("Invalid log type");
	}
	std::lock_guard<std::mutex> lock(m_mutex);
	std::cout << logTypeIter->second << str << std::endl;
};