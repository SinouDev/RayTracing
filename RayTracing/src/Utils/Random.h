#pragma once

#include "Core/Ray.h"

#include "Math.h"

#include <cstdlib>

#if defined(USE_WALNUT_RANDOM)
#include "Walnut/Random.h"
#endif

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
		static __forceinline void __fastcall Init()
		{
			//s_Random.Init();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline double __fastcall RandomDouble()
		{
			// Returns a random real in [0,1).
			//float a = s_Random.Float();
			
#if defined(USE_WALNUT_RANDOM)
			return s_Random.Float();
#else
			return rand() / (RAND_MAX + 1.0f);
#endif
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static __forceinline double __fastcall RandomDouble(double min, double max)
		{
			// Returns a random real in [min,max).
			return min + (max - min) * RandomDouble();
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline int32_t __fastcall RandomInt()
		{
			return static_cast<int32_t>(RandomDouble());
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static __forceinline int32_t __fastcall RandomInt(int32_t min, int32_t max)
		{
			return static_cast<int32_t>(RandomDouble(min, max));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline float __fastcall RandomFloat()
		{
			return static_cast<float>(RandomDouble());
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static __forceinline float __fastcall RandomFloat(float min, float max)
		{
			return static_cast<float>(RandomDouble(min, max));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <returns></returns>
		template<typename T>
		static __forceinline T __fastcall GetRandom()
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
		static __forceinline T __fastcall GetRandom(T min, T max)
		{
			return static_cast<T>(RandomDouble(min, max));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline Utils::Math::Vec3 __fastcall RandomVec3()
		{
			return { RandomDouble(), RandomDouble(), RandomDouble() };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="min"></param>
		/// <param name="max"></param>
		/// <returns></returns>
		static __forceinline Utils::Math::Vec3 __fastcall RandomVec3(double min, double max)
		{
			return { RandomDouble(min, max), RandomDouble(min,  max), RandomDouble(min, max) };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline Utils::Math::Vec3 __fastcall RandomInUnitSphere()
		{
			//return Random::RandomVec3(-1.0, 1.0);
			while (true)
			{
				Utils::Math::Vec3 p = Random::RandomVec3(-1.0, 1.0);
				if (Utils::Math::Dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline Utils::Math::Vec3 __fastcall RandomInUnitDisk()
		{
			//return Random::RandomVec3(-1.0, 1.0);
			while (true)
			{
				Utils::Math::Vec3 p = Utils::Math::Vec3(Random::RandomDouble(-1.0, 1.0), Random::RandomDouble(-1.0, 1.0), 0.0);
				if (Utils::Math::Dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		static __forceinline Utils::Math::Vec3 __fastcall RandomUnitVector()
		{
			Utils::Math::Vec3 tmp = RandomInUnitSphere();
			return tmp / Utils::Math::Q_Length(tmp);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="normal"></param>
		/// <returns></returns>
		static __forceinline Utils::Math::Vec3 __fastcall RandomInHemisphere(Utils::Math::Vec3& normal)
		{
			Utils::Math::Vec3 in_unit_sphere = RandomInUnitSphere();
			if (Utils::Math::Dot(in_unit_sphere, normal) > 0.0f)
			{
				return in_unit_sphere;
			}
			else
			{
				return -in_unit_sphere;
			}
		}
#if defined(USE_WALNUT_RANDOM)
	private:

		static Walnut::Random s_Random;
#endif

	};
}