#pragma once

#include "glm/glm.hpp"

class HittableObject;

class Ray
{

public:

	using Point3 = glm::vec3;
	using Vec3   = glm::vec3;
	using Color  = glm::vec3;

	Ray() = default;
	Ray(const Point3& origin, const Vec3& direction = Vec3(0.0f), float time = 0.0f, const Color& backgroundColor = Color(0.5f, 0.7f, 1.0f), const Color& backgroundColor1 = Color(1.0f));

	inline const Point3& GetOrigin() const { return m_Origin; }
	inline const Vec3& GetDirection() const { return m_Direction; }
	inline const Color& GetRayBackgroundColor() { return m_RayBackgroundColor; }
	inline const Color& GetRayBackgroundColor1() { return m_RayBackgroundColor1; }
	inline const float GetTime() const { return m_Time; }
	//inline vec3& GetLightDir() { return m_LightDir; }

	Point3 At(float t) const;

	static Color RayColor(const Ray& ray, const Color& backgroundColor, const HittableObject& list, int32_t depth);

private:

	friend Color get_background(const Ray&);

private:

	Point3 m_Origin;
	Vec3 m_Direction;
	float m_Time;
	Color m_RayBackgroundColor;
	Color m_RayBackgroundColor1;
	//vec3 m_LightDir = vec3(1.0f, 10.0f, 3.0f);

};