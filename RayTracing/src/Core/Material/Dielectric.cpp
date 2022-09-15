#include "Dielectric.h"

#include "Core/Object/HittableObject.h"
#include "Utils/Random.h"

using Utils::Math::Vec3;
using Utils::Math::Color4;

Dielectric::Dielectric(float index_of_refraction)
	: m_IndexOfRefraction(index_of_refraction)
{
}

bool Dielectric::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{

	attenuation = Color4(1.0f, 1.0f, 1.0f, 1.0f);
	float refraction_ratio = hitRecord.front_face ? (1.0f/m_IndexOfRefraction) : m_IndexOfRefraction;

	Vec3 unit_direction = Utils::Math::UnitVec(ray.GetDirection());// / Utils::Math::Q_Length(ray.GetDirection());

	float cos_theta = Utils::Math::Min(Utils::Math::Dot(-unit_direction, hitRecord.normal), 1.0f);
	float sin_theta = Utils::Math::Q_Sqrt(1.0f - cos_theta * cos_theta);

	bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
	Vec3 direction;

	if (cannot_refract || Utils::Math::Reflectness(cos_theta, refraction_ratio) > Utils::Random::RandomDouble())
	{
		direction = Utils::Math::Reflect(unit_direction, hitRecord.normal);
	}
	else
	{
		direction = Utils::Math::Refract(unit_direction, hitRecord.normal, refraction_ratio);
	}

	//Vec3 refracted = Utils::Math::Refract(unit_direction, hitRecord.normal, refraction_ratio);

	scattered = Ray(hitRecord.point, direction, ray.GetTime());	

	return true;
}