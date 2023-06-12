#pragma once

#include <cstdint>
#define _USE_32BIT_TIME_T
#include <ctime>

namespace Utils {

	namespace Time {

		struct TimeComponents {
			uint64_t milli_seconds;
			uint64_t seconds;
			uint64_t minutes;
			uint64_t hours;
			uint64_t days;
			uint64_t months;
			uint64_t years;
			std::time_t time;

			explicit operator float()
			{
				return static_cast<float>(time);
			}
		};

		static TimeComponents GetTimeComponents(std::time_t time)
		{
#ifdef NDEBUG // TODO: fix this nullptr bug when using debug configuration
			std::tm* st = localtime(&time);
			return { time % 1000ULL,
					 time / 1000ULL % 60ULL,
					 time / 60000ULL % 60ULL,
					 time / 3600000ULL % 24ULL,
					 (uint64_t)st->tm_mday,
					 (uint64_t)st->tm_mon,
					 (uint64_t)st->tm_year,
					 time };
#else
			return { time % 1000ULL,
					 time / 1000ULL % 60ULL,
					 time / 60000ULL % 60ULL,
					 time / 3600000ULL % 24ULL };
#endif

		}

		static void GetTimeComponents(TimeComponents& components, std::time_t time)
		{
			std::tm* st = std::localtime(&time);
			components.milli_seconds = time % 1000ULL;
			components.seconds = time / 1000ULL % 60ULL;
			components.minutes = time / 60000ULL % 60ULL;
			components.hours = time / 3600000ULL % 24ULL;
			components.days = (uint64_t)st->tm_mday;
			components.months = (uint64_t)st->tm_mon;
			components.years = (uint64_t)st->tm_year;
			components.time = time;
		}

		template<typename _Ty>
		static std::time_t ToTime_t(_Ty)
		{
			static_assert(false, "");
		}

		template<>
		static std::time_t ToTime_t(float t)
		{
			return static_cast<std::time_t>(t);
		}

	}
}