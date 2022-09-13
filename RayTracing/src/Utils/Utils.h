#pragma once

#include "Core/AABB.h"
#include "Core/Ray.h"

#include "glm/glm.hpp"

#include <limits>

namespace Utils {

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

	constexpr static float infinity = std::numeric_limits<float>::infinity();

}