#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Utils/Time.h"

#include <filesystem>
#include <iostream>
#include <chrono>

#include "Application/RayTracingApplication.cuh"

#define ENABLE_TEST 0

#if ENABLE_TEST
void test();
#endif

void generate_name(const std::string& path, const std::string& extention, std::string& name);

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{

	Walnut::ApplicationSpecification spec;
	spec.Name = "Ray Tracing";

	Walnut::Application* app = new Walnut::Application(spec);

#if ENABLE_TEST
	test();
	return app;
#endif
	auto exLayer = app->PushAndGetLayer<RayTracingLayer>();

	app->SetMenubarCallback([app, exLayer]()
		{
			std::string path = "Screenshots";

			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Save ppm", "Ctrl + S", nullptr, (bool)exLayer->GetCudaRenderer().GetActiveScene()))
				{
					std::string name;
					std::string ppmPath = path + "/ppm";
					//generate_name(ppmPath, std::string("ppm"), name);
					//system((std::string("mkdir ") + ppmPath).c_str());
					std::filesystem::create_directory(ppmPath);
					exLayer->SavePPM(name.c_str());
				}
				if (ImGui::MenuItem("Save png", "Ctrl + S", nullptr, (bool)exLayer->GetCudaRenderer().GetActiveScene()))
				{
					std::string name;
					//generate_name(path, std::string("png"), name);
					//system((std::string("mkdir ") + path).c_str());
					std::filesystem::create_directory(path);
					exLayer->SavePNG(name.c_str());
				}
				if (ImGui::MenuItem("Exit"))
				{
					app->Close();
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("View"))
			{
				ImGui::Checkbox("Always show object description", &exLayer->GetAlwaysShowDescInObjectList());
				ImGui::EndMenu();
			}
		});
	return app;
}

void generate_name(const std::string& path, const std::string& extention, std::string& name)
{
	auto clock_now = std::chrono::system_clock::now();
	Utils::Time::TimeComponents t = Utils::Time::GetTimeComponents(std::chrono::time_point_cast<std::chrono::milliseconds>(clock_now).time_since_epoch().count());
	//auto t = std::chrono::high_resolution_clock::now();
	//
	//auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(t);
	//auto minutes = std::chrono::time_point_cast<std::chrono::minutes>(t);
	//auto hours   = std::chrono::time_point_cast<std::chrono::hours>(t);
	//
	//auto s = seconds.time_since_epoch().count() % 60;
	//auto m = (minutes.time_since_epoch().count() + s) % 60;
	//auto h = (hours.time_since_epoch().count() + m);

	char buffer[200];;
	memset(buffer, 0, 200);

	sprintf_s(buffer, "%s/snapshot %02u-%02u-%02u %llu.%s", path.c_str(), (uint32_t)t.hours, (uint32_t)t.minutes, (uint32_t)t.seconds, t.time, extention.c_str());
	//= path + "snapshot " + std::to_string(t.hours) + "-" + std::to_string(t.minutes) + "-" + std::to_string(t.seconds) + " " + std::to_string(t.time) + "." + extention;

	std::cout << "Saving file: \"" << buffer << "\"\n";
	name = buffer;
}

#if ENABLE_TEST
#include "ftl/task_counter.h"
#include "ftl/task_scheduler.h"

#include <assert.h>
#include <stdint.h>

struct NumberSubset {
	uint64_t start;
	uint64_t end;

	uint64_t total;
};

void AddNumberSubset(ftl::TaskScheduler* taskScheduler, void* arg) {
	(void)taskScheduler;
	NumberSubset* subset = reinterpret_cast<NumberSubset*>(arg);

	subset->total = 0;

	while (subset->start != subset->end) {
		subset->total += subset->start;
		++subset->start;
	}

	subset->total += subset->end;
}

/**
 * Calculates the value of a triangle number by dividing the additions up into tasks
 *
 * A triangle number is defined as:
 *         Tn = 1 + 2 + 3 + ... + n
 *
 * The code is checked against the numerical solution which is:
 *         Tn = n * (n + 1) / 2
 */
void test()
{
	// Create the task scheduler and bind the main thread to it
	ftl::TaskScheduler taskScheduler;
	taskScheduler.Init();

	// Define the constants to test
	constexpr uint64_t triangleNum = 47593243ULL;
	constexpr uint64_t numAdditionsPerTask = 10000ULL;
	constexpr uint64_t numTasks = (triangleNum + numAdditionsPerTask - 1ULL) / numAdditionsPerTask;

	// Create the tasks
	// FTL allows you to create Tasks on the stack.
	// However, in this case, that would cause a stack overflow
	ftl::Task* tasks = new ftl::Task[numTasks];
	NumberSubset* subsets = new NumberSubset[numTasks];
	uint64_t nextNumber = 1ULL;

	for (uint64_t i = 0ULL; i < numTasks; ++i) {
		NumberSubset* subset = &subsets[i];

		subset->start = nextNumber;
		subset->end = nextNumber + numAdditionsPerTask - 1ULL;
		if (subset->end > triangleNum) {
			subset->end = triangleNum;
		}

		tasks[i] = { AddNumberSubset, subset };

		nextNumber = subset->end + 1;
	}

	// Schedule the tasks
	ftl::TaskCounter counter(&taskScheduler);
	taskScheduler.AddTasks(numTasks, tasks, ftl::TaskPriority::Normal, &counter);

	// FTL creates its own copies of the tasks, so we can safely delete the memory
	delete[] tasks;

	// Wait for the tasks to complete
	taskScheduler.WaitForCounter(&counter);

	// Add the results
	uint64_t result = 0ULL;
	for (uint64_t i = 0; i < numTasks; ++i) {
		result += subsets[i].total;
	}

	// Test
	assert(triangleNum * (triangleNum + 1ULL) / 2ULL == result);

	// Cleanup
	delete[] subsets;

	// The destructor of TaskScheduler will shut down all the worker threads
	// and unbind the main thread
}

#endif