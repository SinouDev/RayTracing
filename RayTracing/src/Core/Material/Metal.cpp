#include "Metal.h"

#include "Core/Object/HittableObject.h"
#include "Core/Random.h"
#include "Core/Utils.h"

Metal::Metal(color& color, float fuzz)
    : m_Albedo(color), m_Fuzz(fuzz < 1.0f ? fuzz : 1.0f)
{
}

bool Metal::Scatter(const Ray& ray, const HitRecord& hitRecord, color& attenuation, Ray& scattered) const
{

    vec3 reflected = glm::reflect(Utils::UnitVec(ray.GetDirection()), hitRecord.normal);

    scattered = Ray(hitRecord.point, reflected + m_Fuzz * Random::RandomInUnitSphere(), ray.GetTime());

    attenuation = m_Albedo;

    return glm::dot(scattered.GetDirection(), hitRecord.normal) > 0.0f;
}

ShinyMetal::ShinyMetal(color& color)
    : Metal(color, 1.0f)
{
    m_Fuzz = 0.0f;
}