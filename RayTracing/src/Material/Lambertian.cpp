#include "Lambertian.h"

#include "../Object/HittableObject.h"
#include "../Random.h"

Lambertian::Lambertian(color& color)
    : m_Albedo(color)
{
}

bool Lambertian::Scatter(Ray& ray, HitRecord& hitRecord, color& attenuation, Ray& scattered) const
{
    vec3 scatter_direction = hitRecord.normal - Walnut::Random::InUnitSphere();

    if (near_zero(scatter_direction))
    {
        scatter_direction = hitRecord.normal;
    }

    scattered = Ray(hitRecord.point, scatter_direction);
    attenuation = m_Albedo;
    return true;
}

bool Lambertian::near_zero(vec3& vector)
{
    vec3 tmp = glm::abs(vector);
    const auto s = 1e-8;
    return (tmp.x < s) && (tmp.y < s) && (tmp.z < s);
}