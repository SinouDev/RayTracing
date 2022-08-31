#include "Metal.h"

#include "../Object/HittableObject.h"
#include "../Random.h"

Metal::Metal(color& color, float fuzz)
    : m_Albedo(color), m_Fuzz(fuzz < 1.0f ? fuzz : 1.0f)
{
}

bool Metal::Scatter(Ray& ray, HitRecord& hitRecord, color& attenuation, Ray& scattered) const
{

    vec3 reflected = glm::reflect(ray.GetDirection() / glm::length(ray.GetDirection()), hitRecord.normal);

    scattered = Ray(hitRecord.point, reflected + m_Fuzz * Random::RandomInUnitSphere());

    attenuation = m_Albedo;

    return glm::dot(scattered.GetDirection(), hitRecord.normal) > 0.0f;
}

ShinyMetal::ShinyMetal(color& color)
    : Metal(color, 1.0f)
{
    m_Fuzz = 0.0f;
}