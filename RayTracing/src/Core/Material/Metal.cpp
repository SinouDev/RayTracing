#include "Metal.h"

#include "Core/Object/HittableObject.h"
#include "Core/Utils.h"
#include "Core/Texture/SolidColorTexture.h"

Metal::Metal(Color& color, float fuzz)
    : m_Albedo(std::make_shared<SolidColorTexture>(color)), m_Fuzz(fuzz < 1.0f ? fuzz : 1.0f)
{
}

Metal::Metal(std::shared_ptr<Texture>& texture, float fuzz)
    : m_Albedo(texture), m_Fuzz(fuzz < 1.0f ? fuzz : 1.0f)
{
}

bool Metal::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{

    Vec3 reflected = glm::reflect(Utils::UnitVec(ray.GetDirection()), hitRecord.normal);

    scattered = Ray(hitRecord.point, reflected + m_Fuzz * Utils::Random::RandomInUnitSphere(), ray.GetTime());

    attenuation = m_Albedo->ColorValue(hitRecord.coord, hitRecord.point);

    return glm::dot(scattered.GetDirection(), hitRecord.normal) > 0.0f;
}

ShinyMetal::ShinyMetal(Color& color)
    : Metal(color, 1.0f)
{
    m_Fuzz = 0.0f;
}

ShinyMetal::ShinyMetal(std::shared_ptr<Texture>& texture)
    : Metal(texture, 1.0f)
{
}
