#pragma once

#if !defined(AABB_CLASS_DEFINED)
#include "Core/AABB.h"
#endif

#if !defined(RAY_CLASS_DEFINED)
#include "Core/Ray.h"
#endif

#if !defined(MATH_UTILS_DEFINED)
#include "Utils/Math.h"
#endif

/// <summary>
/// 
/// </summary>
namespace Utils {

	/// <summary>
	/// 
	/// </summary>
	/// <param name="box0"></param>
	/// <param name="box1"></param>
	/// <returns></returns>
	static inline AABB SurroundingBox(const AABB& box0, const AABB& box1)
	{
		// changed to small_b to not conflict with "#define small char" defined in <rpcndr.h>
		Math::Vec3 small_b(glm::min(box0.GetMin(), box1.GetMin()));
		Math::Vec3 big_b(glm::max(box0.GetMax(), box1.GetMax()));

		return AABB(small_b, big_b);
	}

}