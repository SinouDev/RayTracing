#include "Dielectric.h"

#include "Core/Object/HittableObject.h"
#include "Utils/Utils.h"
#include "Utils/Random.h"

Dielectric::Dielectric(float index_of_refraction)
	: m_IndexOfRefraction(index_of_refraction)
{
}

bool Dielectric::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{

	attenuation = Color4(1.0f, 1.0f, 1.0f, 1.0f);
	float refraction_ratio = hitRecord.front_face ? (1.0f/m_IndexOfRefraction) : m_IndexOfRefraction;

	Vec3 unit_direction = Utils::UnitVec(ray.GetDirection());// / glm::length(ray.GetDirection());

	float cos_theta = glm::min(glm::dot(-unit_direction, hitRecord.normal), 1.0f);
	float sin_theta = glm::sqrt(1.0f - cos_theta * cos_theta);

	bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
	Vec3 direction;

	if (cannot_refract || reflectness(cos_theta, refraction_ratio) > Utils::Random::RandomDouble())
	{
		direction = glm::reflect(unit_direction, hitRecord.normal);
	}
	else
	{
		direction = glm::refract(unit_direction, hitRecord.normal, refraction_ratio);
	}

	//vec3 refracted = refract(unit_direction, hitRecord.normal, refraction_ratio);

	scattered = Ray(hitRecord.point, direction, ray.GetTime());	

	return true;
}

float Dielectric::reflectness(float cosine, float ref_index)
{
	float r0 = (1.0f - ref_index) / (1.0f + ref_index);
	r0 = r0*r0;
	return static_cast<float>(r0 + (1.0f - r0) * glm::pow(1.0f - cosine, 5));
}
