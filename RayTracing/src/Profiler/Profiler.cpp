#define PROFILER_MACRO_PROFILE_FORMATTED

#include "Profiler.h"

#include <string>
#include <algorithm>
#include <thread>
#include <Windows.h>
#include <sysinfoapi.h>
#include <processthreadsapi.h>
#include <iostream>

#ifdef _WIN32
#define GET_PROCESS_ID GetCurrentProcessId
#else
#define GET_PROCESS_ID ::getpid
#endif

std::ostream& operator<<(std::ostream& stream, const MachineInfo& info)
{
	stream << "{\"cpu_brand\": \"" << info.cpu_brand << "\", \"cpu_family\": \"" << info.cpu_family << "\", \"cpu_model\": \"" << info.cpu_model << "\", \"cpu_stepping\": \"" << info.cpu_stepping << "\"}";
	return stream;
}

std::ostream& operator<<(std::ostream& stream, const ProfilerSession& session)
{
	stream << "{\"session_name\": \"" << session.session_name << "\"}";
	return stream;
}

void MachineInfo::LoadSystemInfo() const
{
	//**
}

void Profiler::Begin(const std::string& session_name, const std::string& file_path) 
{
	m_CurrentProfilerSession = new ProfilerSession{ session_name };
	m_JsonProfilerOutput.open(file_path);
	WriteHeader();
}

void Profiler::End()
{
	WriteFooter();
	m_JsonProfilerOutput.close();
	delete m_CurrentProfilerSession;
	m_CurrentProfilerSession = nullptr;
	m_ProfileCount = 0;
}

void Profiler::Write(const ProfilerResult& result)
{
	if (m_ProfileCount++ > 1)
		m_JsonProfilerOutput << "," << MACRO_PROFILE_LINE_1;

	std::string name = result.session_name;
	std::replace(name.begin(), name.end(), '\"', '\'');

	m_JsonProfilerOutput << MACRO_PROFILE_LINE_3 << "{" << MACRO_PROFILE_LINE_4 << "\"args\": " << "{}" << ",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"cat\": \"" << result.category << "\",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"dur\": " << (result.end_time - result.start_time) << ",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"name\": \"" << result.session_name << "\",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"ph\": \"" << "X" << "\",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"pid\":" << result.process_id << ",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"tid\": " << result.thread_id << ",";
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_4 << "\"ts\": " << result.start_time;
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_3 << "}";

	m_JsonProfilerOutput.flush();

}

void Profiler::WriteHeader()
{
	MachineInfo info;
	m_JsonProfilerOutput << "{" << MACRO_PROFILE_LINE_2 << "\"otherData\": " << *m_CurrentProfilerSession << ", " << MACRO_PROFILE_LINE_2 << "\"metaData\": " << info << ", " << MACRO_PROFILE_LINE_2 << "\"traceEvents\": [";
	m_JsonProfilerOutput.flush();
}

void Profiler::WriteFooter()
{
	m_JsonProfilerOutput << MACRO_PROFILE_LINE_2 << "]" << MACRO_PROFILE_LINE_1 << "}";
	m_JsonProfilerOutput.flush();
}

Profiler& Profiler::Get()
{
	static Profiler profiler;
	return profiler;
}

uint64_t ProfilerTimer::ToLong(std::chrono::time_point<std::chrono::high_resolution_clock> time)
{
	return std::chrono::time_point_cast<std::chrono::microseconds>(time).time_since_epoch().count();
}

ProfilerTimer::ProfilerTimer(const char* name)
	: m_ProfilerName(name)
{
	m_StartTime = std::chrono::high_resolution_clock::now();
}

ProfilerTimer::~ProfilerTimer()
{
	if (!m_Stopped)
		Stop();
}

void ProfilerTimer::Stop()
{
	auto endTime = std::chrono::high_resolution_clock::now();

	uint64_t start = ToLong(m_StartTime);
	uint64_t end = ToLong(endTime);

	uint32_t threadId = std::hash<std::thread::id>{}(std::this_thread::get_id());

	uint32_t processId = GET_PROCESS_ID();

	Profiler::Get().Write({ m_ProfilerName, start, end, threadId, processId });
	m_Stopped = true;
}