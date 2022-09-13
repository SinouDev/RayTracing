#pragma once

#include "Core/AABB.h"
#include "Core/Ray.h"

#include "glm/glm.hpp"

#include <limits>

namespace Utils {

	static AABB SurroundingBox(const AABB& box0, const AABB& box1)
	{
		// changed to small_b to not conflict with "#define small char" defined in <rpcndr.h>
		glm::vec3 small_b(glm::min(box0.GetMin(), box1.GetMin()));
		glm::vec3 big_b(glm::max(box0.GetMax(), box1.GetMax()));

		return AABB(small_b, big_b);
	}

	template<typename T>
	static T UnitVec(T t)
	{
		return t / glm::length(t);
	}

	constexpr static float infinity = std::numeric_limits<float>::infinity();

}