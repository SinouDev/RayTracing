#include "ConstantMedium.h"

#include "Core/AABB.h"
#include "Core/Material/Isotropic.h"

#include "Utils/Random.h"

using Utils::Math::Color3;
using Utils::Math::Vec3;

ConstantMedium::ConstantMedium(std::shared_ptr<HittableObject>& object, float d, std::shared_ptr<Texture>& texture)
	: m_Boundary(object), m_NegInvDensity(-1.0f / d), m_PhaseFunction(std::make_shared<Isotropic>(texture))
{
	m_Name = "ConstantMedium";
	m_FeatureObject = true;
}

ConstantMedium::ConstantMedium(std::shared_ptr<HittableObject>& object, float d, Color3 color)
	: m_Boundary(object), m_NegInvDensity(-1.0f / d), m_PhaseFunction(std::make_shared<Isotropic>(color))
{
	m_Name = "ConstantMedium";
	m_FeatureObject = true;
}

bool ConstantMedium::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	if (!m_Hittable)
		return false;

	HitRecord hitRecord1, hitRecord2;


	if (!m_Boundary->Hit(ray, -Utils::Math::infinity, Utils::Math::infinity, hitRecord1))
		return false;

	if (!m_Boundary->Hit(ray, hitRecord1.t + 0.0001f, Utils::Math::infinity, hitRecord2))
		return false;

	hitRecord1.t = Utils::Math::Max(hitRecord1.t, min);
	hitRecord2.t = Utils::Math::Min(hitRecord2.t, max);

	if (hitRecord1.t >= hitRecord2.t)
		return false;

	hitRecord1.t = Utils::Math::Max(hitRecord1.t, 0.0f);

	const float rayLength = Utils::Math::Q_Length(ray.GetDirection());
	const float distanceInsideBoundary = (hitRecord2.t - hitRecord1.t) * rayLength;
	const float hitDistance = m_NegInvDensity * Utils::Math::Log(Utils::Random::RandomFloat());

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
	if (!m_Hittable)
		return false;

	return m_Boundary->BoundingBox(_time0, _time1, output_box);
}
