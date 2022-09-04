#pragma once

#include "glm/glm.hpp"

class Ray;

class AABB
{

	using Point3 = glm::vec3;

public:
	AABB();
	AABB(const Point3& a, const Point3& b);

	bool Hit(const Ray& ray, float t_min, float t_max) const;

	inline Point3 GetMin() const { return m_Minimum; }
	inline Point3 GetMax() const { return m_Maximum; }

private:

	Point3 m_Minimum;
	Point3 m_Maximum;

};