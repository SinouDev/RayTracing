#include "Lambertian.h"

#include "Core/Object/HittableObject.h"
#include "Core/Texture/SolidColorTexture.h"

#include "Utils/Random.h"
#include "Utils/Color.h"

using Utils::Math::Color3;
using Utils::Math::Color4;
using Utils::Math::Vec3;

Lambertian::Lambertian(Color3& color)
    : Lambertian(Color4(color, 1.0f))
{
}

Lambertian::Lambertian(Color4& color)
    : m_Albedo(std::make_shared<SolidColorTexture>(color))
{
}

Lambertian::Lambertian(std::shared_ptr<Texture>& texture)
    : m_Albedo(texture)
{
}

bool Lambertian::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{
    Vec3 scatter_direction = hitRecord.normal - Utils::Random::RandomInUnitSphere();

    if (Utils::Math::NearZero(scatter_direction))
    {
        scatter_direction = hitRecord.normal;
    }

    scattered = Ray(hitRecord.point, scatter_direction, ray.GetTime());

    Color4 attenuation0 = m_Albedo->ColorValue(hitRecord.coord, hitRecord.point);

    if (m_Material)
    {
        Color4 attenuation1;
        if(m_Material->Scatter(ray, hitRecord, attenuation1, scattered))
            attenuation = Utils::Color::Vec4ToRGBABlendColor(attenuation1, attenuation0);
    }
    else
        attenuation = attenuation0;

    return true;
}