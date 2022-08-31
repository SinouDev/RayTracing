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
	Ray(const point3& origin, const vec3& direction, const color& backgroundColor, const color& backgroundColor1);

	inline point3& GetOrigin() { return m_Origin; }
	inline vec3& GetDirection() { return m_Direction; }
	inline color& GetRayBackgroundColor() { return m_RayBackgroundColor; }
	inline color& GetRayBackgroundColor1() { return m_RayBackgroundColor1; }
	inline vec3& GetLightDir() { return m_LightDir; }

	point3 At(float t) const;

	static color RayColor(Ray& ray, HittableObjectList& list, int32_t depth);

private:

	point3 m_Origin;
	vec3 m_Direction;
	color m_RayBackgroundColor;
	color m_RayBackgroundColor1;
	vec3 m_LightDir = vec3(1.0f, 10.0f, 3.0f);

};