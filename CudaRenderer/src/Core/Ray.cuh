#pragma once

#include "gmath.cuh"

namespace SGOL {
	struct Ray {
		__device__ __host__ Ray()
			: Ray(glm::vec3{0.0f}, glm::vec3{0.0f}, 0.0f)
		{}

		__device__ __host__ Ray(glm::vec3 origin, glm::vec3 direction, float time)
			: origin(origin), direction(direction), time(time)
		{}

		__device__ __host__ Ray(const Ray& ray)
			: origin(ray.origin), direction(ray.direction), time(ray.time)
		{}

		glm::vec3 origin{0.0f, 0.0f, -1.0f};
		glm::vec3 direction{0.0f};
		float time;

	};
}