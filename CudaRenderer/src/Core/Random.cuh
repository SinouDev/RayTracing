#pragma once

#include <stdint.h>
#include <math.h>
#include <random>

#include "gmath.cuh"

namespace SGOL {
	class Random
	{
	public:

		static __forceinline double __fastcall RandomDouble()
		{
			return rand() / (RAND_MAX + 1.0f);

		}

		static __forceinline double __fastcall RandomDouble(double min, double max)
		{
			return min + (max - min) * RandomDouble();
		}

		static __forceinline int32_t __fastcall RandomInt()
		{
			return static_cast<int32_t>(RandomDouble());
		}

		static __forceinline int32_t __fastcall RandomInt(int32_t min, int32_t max)
		{
			return static_cast<int32_t>(RandomDouble(min, max));
		}

		static __forceinline float __fastcall RandomFloat()
		{
			return static_cast<float>(RandomDouble());
		}

		static __forceinline float __fastcall RandomFloat(float min, float max)
		{
			return static_cast<float>(RandomDouble(min, max));
		}

		template<typename T>
		static __forceinline T __fastcall GetRandom()
		{
			return static_cast<T>(RandomDouble());
		}

		template<typename T>
		static __forceinline T __fastcall GetRandom(T min, T max)
		{
			return static_cast<T>(RandomDouble(min, max));
		}

		static __forceinline glm::vec3 __fastcall RandomVec3()
		{
			return { RandomDouble(), RandomDouble(), RandomDouble() };
		}

		static __forceinline glm::vec3 __fastcall RandomVec3(double min, double max)
		{
			return { RandomDouble(min, max), RandomDouble(min,  max), RandomDouble(min, max) };
		}

		static __forceinline glm::vec3 __fastcall RandomInUnitSphere()
		{
			while (true)
			{
				glm::vec3 p = Random::RandomVec3(-1.0, 1.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		static __forceinline glm::vec3 __fastcall RandomInUnitDisk()
		{
			while (true)
			{
				glm::vec3 p = glm::vec3(Random::RandomDouble(-1.0, 1.0), Random::RandomDouble(-1.0, 1.0), 0.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		static __forceinline glm::vec3 __fastcall RandomUnitVector()
		{
			glm::vec3 tmp = RandomInUnitSphere();
			return tmp / glm::fastLength(tmp);
		}

		static __forceinline glm::vec3 __fastcall RandomInHemisphere(glm::vec3& normal)
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
	};
}