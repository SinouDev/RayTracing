#include "Translate.h"

#include "Core/Ray.h"
#include "Core/AABB.h"

Translate::Translate(std::shared_ptr<HittableObject>& object, const Vec3& displacement)
	: m_Object(object), m_Offset(displacement)
{
}

bool Translate::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	Ray movedR(ray.GetOrigin() - m_Offset, ray.GetDirection(), ray.GetTime());
	if(!m_Object->Hit(movedR, min, max, hitRecord))
		return false;

	hitRecord.point += m_Offset;
	hitRecord.set_face_normal(movedR, hitRecord.normal);

	return true;
}

bool Translate::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	if(!m_Object->BoundingBox(_time0, _time1, output_box))
		return false;

	output_box = AABB(output_box.GetMin() + m_Offset, output_box.GetMax() + m_Offset);

	return true;
}
