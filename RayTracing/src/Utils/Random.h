#pragma once

#include "Core/Ray.h"

#include "Walnut/Random.h"

#include "Math.h"

#include <cstdlib>

/// <summary>
/// 
/// </summary>
namespace Utils {

	/// <summary>
	/// 
	/// </summary>
	class Random
	{
	public:

		/// <summary>
		/// 
		/// </summary>
		static inline void Init()
		{
			s_Random.Init();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline double RandomDouble()
		{
			// Returns a random real in [0,1).
			//float a = s_Random.Float();
			return s_Random.Float();// rand() / (RAND_MAX + 1.0f);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static inline double RandomDouble(double min, double max)
		{
			// Returns a random real in [min,max).
			return min + (max - min) * RandomDouble();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline int32_t RandomInt()
		{
			return static_cast<int32_t>(RandomDouble());
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static inline int32_t RandomInt(int32_t min, int32_t max)
		{
			return static_cast<int32_t>(RandomDouble(min, max));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline float RandomFloat()
		{
			return static_cast<float>(RandomDouble());
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static inline float RandomFloat(float min, float max)
		{
			return static_cast<float>(RandomDouble(min, max));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <returns></returns>
		template<typename T>
		static inline T GetRandom()
		{
			return static_cast<T>(RandomDouble());
		}

		/// <summary>
		/// 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		template<typename T>
		static inline T GetRandom(T min, T max)
		{
			return static_cast<T>(RandomDouble(min, max));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline Utils::Math::Vec3 RandomVec3()
		{
			return { RandomDouble(), RandomDouble(), RandomDouble() };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static inline Utils::Math::Vec3 RandomVec3(double min, double max)
		{
			return { RandomDouble(min, max), RandomDouble(min,  max), RandomDouble(min, max) };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline Utils::Math::Vec3 RandomInUnitSphere()
		{
			//return Random::RandomVec3(-1.0, 1.0);
			while (true)
			{
				Utils::Math::Vec3 p = Random::RandomVec3(-1.0, 1.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline Utils::Math::Vec3 RandomInUnitDisk()
		{
			//return Random::RandomVec3(-1.0, 1.0);
			while (true)
			{
				Utils::Math::Vec3 p = Utils::Math::Vec3(Random::RandomDouble(-1.0, 1.0), Random::RandomDouble(-1.0, 1.0), 0.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static inline Utils::Math::Vec3 RandomUnitVector()
		{
			Utils::Math::Vec3 tmp = RandomInUnitSphere();
			return tmp / Utils::Math::Q_Length(tmp);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="normal"></param>
		/// <returns></returns>
		static inline Utils::Math::Vec3 RandomInHemisphere(Utils::Math::Vec3& normal)
		{
			Utils::Math::Vec3 in_unit_sphere = RandomInUnitSphere();
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
}