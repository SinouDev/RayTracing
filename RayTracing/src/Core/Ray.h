#pragma once

#include "glm/glm.hpp"

class HittableObjectList;

class Ray
{

public:

	using point3 = glm::vec3;
	using vec3   = glm::vec3;
	using color  = glm::vec3;

	Ray() = delete;
	Ray(const point3& origin, const vec3& direction = vec3(0.0f), float time = 0.0f, const color& backgroundColor = color(0.5f, 0.7f, 1.0f), const color& backgroundColor1 = color(1.0f));

	inline const point3& GetOrigin() const { return m_Origin; }
	inline const vec3& GetDirection() const { return m_Direction; }
	inline const color& GetRayBackgroundColor() { return m_RayBackgroundColor; }
	inline const color& GetRayBackgroundColor1() { return m_RayBackgroundColor1; }
	inline const float GetTime() const { return m_Time; }
	//inline vec3& GetLightDir() { return m_LightDir; }

	point3 At(float t) const;

	static color RayColor(const Ray& ray, const HittableObjectList& list, int32_t depth);

private:

	point3 m_Origin;
	vec3 m_Direction;
	float m_Time;
	color m_RayBackgroundColor;
	color m_RayBackgroundColor1;
	//vec3 m_LightDir = vec3(1.0f, 10.0f, 3.0f);

};