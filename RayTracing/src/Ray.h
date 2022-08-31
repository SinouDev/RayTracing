#pragma once

#include "glm/glm.hpp"

class HittableObjectList;

class Ray
{

public:

	using point3 = glm::vec3;
	using vec3   = glm::vec3;
	using color  = glm::vec3;

	Ray() = default;
	Ray(const point3& origin, const vec3& direction);

	point3& GetOrigin() { return m_Origin; }
	vec3& GetDirection() { return m_Direction; }

	point3 At(float t) const;

	static color RayColor(Ray& ray, HittableObjectList& list, int32_t depth);

private:

	point3 m_Origin;
	vec3 m_Direction;

};