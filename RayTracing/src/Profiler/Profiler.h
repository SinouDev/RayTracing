#pragma once

#include <fstream>
#include <chrono>

struct ProfilerResult {
	std::string session_name;
	uint64_t start_time;
	uint64_t end_time;
	uint32_t thread_id;
	uint32_t process_id;
	std::string category = "function";
};

class MachineInfo {
	std::string cpu_brand;
	std::string cpu_family;
	std::string cpu_model;
	std::string cpu_stepping;

	friend std::ostream& operator<<(std::ostream& stream, const MachineInfo& info);

public:
	void LoadSystemInfo() const;

};

struct ProfilerSession {
	std::string session_name;

	friend std::ostream& operator<<(std::ostream& stream, const ProfilerSession& session);

};

class Profiler {
	std::ofstream m_JsonProfilerOutput;
	ProfilerSession* m_CurrentProfilerSession = nullptr;
	unsigned int m_ProfileCount = 0;

	Profiler() = default;

public:
	
	void Begin(const std::string& session_name, const std::string& file_path = "profiler_trace.json");

	void End();

	void Write(const ProfilerResult& result);

	void WriteHeader();

	void WriteFooter();

	static Profiler& Get();

};

class ProfilerTimer {
	const char* m_ProfilerName;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime;
	bool m_Stopped = false;

	static uint64_t ToLong(std::chrono::time_point<std::chrono::high_resolution_clock> time);

public:
	explicit ProfilerTimer(const char* name);

	~ProfilerTimer();

	void Stop();

};

#define PROFILE_SCOPE(name) ProfilerTimer profile##__LINE__(name)
#define PROFILE_CALL() PROFILE_SCOPE(__FUNCSIG__)

#ifdef PROFILER_MACRO_PROFILE_FORMATTED

constexpr auto MACRO_PROFILE_LINE_1 = "\n";
constexpr auto MACRO_PROFILE_LINE_2 = "\n\t\t";
constexpr auto MACRO_PROFILE_LINE_3 = "\n\t\t\t\t";
constexpr auto MACRO_PROFILE_LINE_4 = "\n\t\t\t\t\t\t";

#else

constexpr auto MACRO_PROFILE_LINE_1 = "";
constexpr auto MACRO_PROFILE_LINE_2 = "";
constexpr auto MACRO_PROFILE_LINE_3 = "";
constexpr auto MACRO_PROFILE_LINE_4 = "";

#endif