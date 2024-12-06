
#include <mutex>
#include <string>
#include <unordered_map>

class Logger
{
public:
	enum LogType
	{
		Blank,
		Info,
		Error
	};
	void operator()(const LogType &logType, const std::string& str);
private:
	std::mutex m_mutex;
	static std::unordered_map<LogType, std::string> logTypeMap;
};

inline static Logger logger = Logger();