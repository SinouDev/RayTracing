#pragma once

#include "AABB.h"
#include "Ray.h"

#include <cstdlib>

#include "Walnut/Random.h"
#include "glm/glm.hpp"

#include <ctime>

namespace Utils {

	namespace Color {
		enum {
			KeepColorChannel = 0xCC5
		};

		static void RGBAtoColorChannels(uint8_t* buffer, uint32_t rgba)
		{
			buffer[0] = (rgba & 0xFF000000) >> 0x18;
			buffer[1] = (rgba & 0x00FF0000) >> 0x10;
			buffer[2] = (rgba & 0x0000FF00) >> 0x08;
			buffer[3] = (rgba & 0x000000FF);
		}

		static void RGBAtoVec4(glm::vec4& ve, uint32_t rgba)
		{
			ve.a = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
			ve.b = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
			ve.g = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
			ve.r = (float)((rgba & 0x000000FF)) / 255.0f;
		}

		static void RGBAtoVec3(glm::vec3& ve, uint32_t rgba)
		{
			//ve.a = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
			ve.b = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
			ve.g = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
			ve.r = (float)((rgba & 0x000000FF)) / 255.0f;
		}

		static glm::vec4 RGBAtoVec4(uint32_t rgba)
		{
			return {
				(float)((rgba & 0x000000FF)) / 255.0f,
				(float)((rgba & 0x0000FF00) >> 0x08) / 255.0f,
				(float)((rgba & 0x00FF0000) >> 0x10) / 255.0f,
				(float)((rgba & 0xFF000000) >> 0x18) / 255.0f };
		}

		static glm::vec3 RGBAtoVec3(uint32_t rgba)
		{
			return {
				(float)((rgba & 0x000000FF)) / 255.0f,
				(float)((rgba & 0x0000FF00) >> 0x08) / 255.0f,
				(float)((rgba & 0x00FF0000) >> 0x10) / 255.0f };
		}

		static glm::vec4 RGBAtoVec4(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
		{
			return {
				(float)(r) / 255.0f,
				(float)(g) / 255.0f,
				(float)(b) / 255.0f,
				(float)(a) / 255.0f };
		}

		static glm::vec3 RGBAtoVec3(uint8_t r, uint8_t g, uint8_t b)
		{
			return {
				(float)(r) / 255.0f,
				(float)(g) / 255.0f,
				(float)(b) / 255.0f };
		}

		static void RGBAtoColorFloats(float* buffer, uint32_t rgba)
		{
			buffer[0] = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
			buffer[1] = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
			buffer[2] = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
			buffer[3] = (float)((rgba & 0x000000FF)) / 255.0f;
		}


		static uint32_t RGBAtoBRGA(uint32_t rgba)
		{
			return (rgba & 0xFF000000) | (rgba & 0x00FF0000) >> 0x10 | (rgba & 0x0000FF00) << 0x08 | (rgba & 0x000000FF) << 0x08;
		}

		static uint32_t FlipRGBA(uint32_t rgba)
		{
			return (rgba & 0x000000FF) << 0x18 | (rgba & 0x0000FF00) << 0x08 | (rgba & 0x00FF0000) >> 0x08 | (rgba & 0xFF000000) >> 0x18;
		}

		static uint8_t GetAlphaChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0xFF000000) >> 0x18);
		}

		static uint8_t GetBlueChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0x00FF0000) >> 0x10);
		}

		static uint8_t GetGreenChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0x0000FF00) >> 0x08);
		}

		static uint8_t GetRedChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0x000000FF));
		}


		static void SetAlphaChannel(uint32_t* rgba, uint8_t a)
		{
			(*rgba) = ((*rgba) & 0x00FFFFFF) | a << 0x18;
		}

		static void SetBlueChannel(uint32_t* rgba, uint8_t b)
		{
			(*rgba) = ((*rgba) & 0xFF00FFFF) | b << 0x10;
		}

		static void SetGreenChannel(uint32_t* rgba, uint8_t g)
		{
			(*rgba) = ((*rgba) & 0xFFFF00FF) | g << 0x08;
		}

		static void SetRedChannel(uint32_t* rgba, uint8_t r)
		{
			(*rgba) = ((*rgba) & 0xFFFFFF00) | r;
		}

		static void SetColorChannel(uint32_t* rgba, uint16_t r = KeepColorChannel, uint16_t g = KeepColorChannel, uint16_t b = KeepColorChannel, uint16_t a = KeepColorChannel)
		{
			(*rgba) = a != KeepColorChannel ? ((*rgba) & 0x00FFFFFF) | a << 0x18 : (*rgba);
			(*rgba) = b != KeepColorChannel ? ((*rgba) & 0xFF00FFFF) | b << 0x10 : (*rgba);
			(*rgba) = g != KeepColorChannel ? ((*rgba) & 0xFFFF00FF) | g << 0x08 : (*rgba);
			(*rgba) = r != KeepColorChannel ? ((*rgba) & 0xFFFFFF00) | r : (*rgba);
		}


		static uint32_t ColorChannelsToRGBA(uint8_t* colors)
		{
			return colors[0] << 0x18 | colors[1] << 0x10 | colors[2] << 0x08 | colors[3];
		}

		static uint32_t ColorChannelsToRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF)
		{
			return a << 0x18 | b << 0x10 | g << 0x08 | r;
		}

		static uint32_t Vec4ToRGBA(glm::vec4& ve)
		{
			return (uint8_t)(ve.a * 255.0f) << 0x18 | (uint8_t)(ve.b * 255.0f) << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		static uint32_t Vec3ToRGBA(glm::vec3& ve)
		{
			return 0xFF << 0x18 | (uint8_t)(ve.b * 255.0f) << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		static uint32_t Vec2ToRGBA(glm::vec2& ve)
		{
			return 0xFF << 0x18 | 0x00 << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		static uint32_t Vec1ToRGBA(glm::vec1& ve)
		{
			return 0xFF << 0x18 | 0x00 << 0x10 | 0x00 << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		static uint32_t ColorFloatsToRGBA(float r, float g, float b, float a = 1.0f)
		{
			return (uint8_t)(a * 255.0f) << 0x18 | (uint8_t)(b * 255.0f) << 0x10 | (uint8_t)(g * 255.0f) << 0x08 | (uint8_t)(r * 255.0f);
		}

		static glm::vec4 Vec4ToRGBABlendColor(glm::vec4& foreground, glm::vec4& background)
		{
			return glm::vec4((glm::vec4(foreground) * (foreground.a)) + (glm::vec4(background) * (1.0f - foreground.a)));
		}

		static uint32_t RGBABlendColor(uint32_t foreground, uint32_t background)
		{
			glm::vec4 foregroundVec;
			RGBAtoVec4(foregroundVec, foreground);
			//if (foregroundVec.a >= 1.0f)
				//return foreground;
			glm::vec4 backgroundVec;
			RGBAtoVec4(backgroundVec, background);
			return Vec4ToRGBA(Vec4ToRGBABlendColor(foregroundVec, backgroundVec));
		}
	}

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
		};

		static TimeComponents GetTime(std::time_t time)
		{
			std::tm* st = std::localtime(&time);
			return { time % 1000ULL,
					 time /    1000ULL % 60ULL,
					 time /   60000ULL % 60ULL,
					 time / 3600000ULL % 24ULL,
					 (uint64_t) st->tm_mday,
					 (uint64_t) st->tm_mon,
					 (uint64_t) st->tm_year,
					 time};
		}

		static void GetTime(TimeComponents& components, std::time_t time)
		{
			std::tm* st = std::localtime(&time);
			components.milli_seconds = time % 1000ULL;
			components.seconds       = time /    1000ULL % 60ULL;
			components.minutes       = time /   60000ULL % 60ULL;
			components.hours         = time / 3600000ULL % 24ULL;
			components.days			 = (uint64_t)st->tm_mday;
			components.months		 = (uint64_t)st->tm_mon;
			components.years		 = (uint64_t)st->tm_year;
			components.time			 = time;
		}

	}

	class Random
	{
	public:

		static inline void Init()
		{
			s_Random.Init();
		}

		static inline double RandomDouble()
		{
			// Returns a random real in [0,1).
			//float a = s_Random.Float();
			return s_Random.Float();// rand() / (RAND_MAX + 1.0f);
		}

		static inline double RandomDouble(double min, double max)
		{
			// Returns a random real in [min,max).
			return min + (max - min) * RandomDouble();
		}

		static inline int32_t RandomInt()
		{
			return static_cast<int32_t>(RandomDouble());
		}

		static inline int32_t RandomInt(int32_t min, int32_t max)
		{
			return static_cast<int32_t>(RandomDouble(min, max));
		}

		static inline float RandomFloat()
		{
			return static_cast<float>(RandomDouble());
		}

		static inline float RandomFloat(float min, float max)
		{
			return static_cast<float>(RandomDouble(min, max));
		}

		template<typename T>
		static inline T GetRandom()
		{
			return static_cast<T>(RandomDouble());
		}

		template<typename T>
		static inline T GetRandom(T min, T max)
		{
			return static_cast<T>(RandomDouble(min, max));
		}

		static inline glm::vec3 RandomVec3()
		{
			return { RandomDouble(), RandomDouble(), RandomDouble() };
		}

		static inline glm::vec3 RandomVec3(double min, double max)
		{
			return { RandomDouble(min, max), RandomDouble(min,  max), RandomDouble(min, max) };
		}

		static inline glm::vec3 RandomInUnitSphere()
		{
			//return Random::RandomVec3(-1.0, 1.0);
			while (true)
			{
				glm::vec3 p = Random::RandomVec3(-1.0, 1.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		static inline glm::vec3 RandomInUnitDisk()
		{
			//return Random::RandomVec3(-1.0, 1.0);
			while (true)
			{
				glm::vec3 p = glm::vec3(Random::RandomDouble(-1.0, 1.0), Random::RandomDouble(-1.0, 1.0), 0.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		static inline glm::vec3 RandomUnitVector()
		{
			glm::vec3 tmp = RandomInUnitSphere();
			return tmp / glm::length(tmp);
		}

		static inline glm::vec3 RandomInHemisphere(Ray::Vec3& normal)
		{
			glm::vec3 in_unit_sphere = RandomInUnitSphere();
			if (glm::dot(in_unit_sphere, normal) > 0.0f)
			{
				return in_unit_sphere;
			}
			else
			{
				return -in_unit_sphere;
			}
		}

	private:

		static Walnut::Random s_Random;

	};

	static AABB SurroundingBox(const AABB& box0, const AABB& box1)
	{
		glm::vec3 small(glm::min(box0.GetMin(), box1.GetMin()));
		glm::vec3 big(glm::max(box0.GetMax(), box1.GetMax()));

		return AABB(small, big);
	}

	template<typename T>
	static T UnitVec(T t)
	{
		return t / glm::length(t);
	}

}