#include "AABB.h"

#include "Ray.h"
#include <algorithm>

using Utils::Math::Point3;

AABB::AABB()
{
}

AABB::AABB(const Point3& a, const Point3& b)
	: m_Minimum(a), m_Maximum(b)
{
}

bool AABB::Hit(const Ray& ray, float t_min, float t_max) const
{
	for (int i = 0; i < 3; i++)
	{
		//float t0 = Utils::Math::Min((m_Minimum[i] - ray.GetOrigin()[i]) / ray.GetDirection()[i], (m_Maximum[i] - ray.GetOrigin()[i]) / ray.GetDirection()[i]);
		//float t1 = Utils::Math::Max((m_Minimum[i] - ray.GetOrigin()[i]) / ray.GetDirection()[i], (m_Maximum[i] - ray.GetOrigin()[i]) / ray.GetDirection()[i]);

		float invD = 1.0f / ray.GetDirection()[i];
		float t0 = (m_Minimum[i] - ray.GetOrigin()[i]) * invD;
		float t1 = (m_Maximum[i] - ray.GetOrigin()[i]) * invD;
		if (invD < 0.0f)
			std::swap(t0, t1);

		t_min = Utils::Math::Max(t0, t_min);
		t_max = Utils::Math::Min(t1, t_max);
		if (t_max <= t_min)
			return false;
	}
	return true;
}
