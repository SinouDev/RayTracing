#include "Lambertian.h"

#include "Core/Object/HittableObject.h"
#include "Core/Utils.h"
#include "Core/Texture/SolidColorTexture.h"

Lambertian::Lambertian(Color& color)
    : m_Albedo(std::make_shared<SolidColorTexture>(color))
{
}

Lambertian::Lambertian(std::shared_ptr<Texture>& texture)
    : m_Albedo(texture)
{
}

bool Lambertian::Scatter(const Ray& ray, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const
{
    Vec3 scatter_direction = hitRecord.normal - Utils::Random::RandomInUnitSphere();

    if (near_zero(scatter_direction))
    {
        scatter_direction = hitRecord.normal;
    }

    scattered = Ray(hitRecord.point, scatter_direction, ray.GetTime());
    attenuation = m_Albedo->ColorValue(hitRecord.coord, hitRecord.point);
    return true;
}

bool Lambertian::near_zero(Vec3& vector)
{
    Vec3 tmp = glm::abs(vector);
    const auto s = 1e-8;
    return (tmp.x < s) && (tmp.y < s) && (tmp.z < s);
}