#pragma once

#include "glm/glm.hpp"

class Ray;

class AABB
{

	using point3 = glm::vec3;

public:
	AABB();
	AABB(const point3& a, const point3& b);

	bool Hit(const Ray& ray, float t_min, float t_max) const;

	inline point3 GetMin() const { return m_Minimum; }
	inline point3 GetMax() const { return m_Maximum; }

private:

	point3 m_Minimum;
	point3 m_Maximum;

};