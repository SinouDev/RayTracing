#include "ConstantMedium.h"

#include "Core/AABB.h"
#include "Core/Utils.h"
#include "Core/Material/Isotropic.h"

ConstantMedium::ConstantMedium(std::shared_ptr<HittableObject>& object, float d, std::shared_ptr<Texture>& texture)
	: m_Boundary(object), m_NegInvDensity(-1.0f / d), m_PhaseFunction(std::make_shared<Isotropic>(texture))
{
}

ConstantMedium::ConstantMedium(std::shared_ptr<HittableObject>& object, float d, Color color)
	: m_Boundary(object), m_NegInvDensity(-1.0f / d), m_PhaseFunction(std::make_shared<Isotropic>(color))
{
}

bool ConstantMedium::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	HitRecord hitRecord1, hitRecord2;


	if (!m_Boundary->Hit(ray, -Utils::infinity, Utils::infinity, hitRecord1))
		return false;

	if (!m_Boundary->Hit(ray, hitRecord1.t + 0.0001f, Utils::infinity, hitRecord2))
		return false;

	hitRecord1.t = glm::max(hitRecord1.t, min);
	hitRecord2.t = glm::min(hitRecord2.t, max);

	if (hitRecord1.t >= hitRecord2.t)
		return false;

	hitRecord1.t = glm::max(hitRecord1.t, 0.0f);

	const float rayLength = glm::length(ray.GetDirection());
	const float distanceInsideBoundary = (hitRecord2.t - hitRecord1.t) * rayLength;
	const float hitDistance = m_NegInvDensity * glm::log(Utils::Random::RandomFloat());

	if (hitDistance > distanceInsideBoundary)
		return false;

	hitRecord.t = hitRecord1.t + hitDistance / rayLength;
	hitRecord.point = ray.At(hitRecord.t);

	hitRecord.normal = Vec3(1.0f, 0.0f, 0.0f);
	hitRecord.front_face = true;
	hitRecord.material_ptr = m_PhaseFunction;

	return true;

}

bool ConstantMedium::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	return m_Boundary->BoundingBox(_time0, _time1, output_box);
}
