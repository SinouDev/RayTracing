#include "DiffuseLight.h"

#include "Core/Texture/SolidColorTexture.h"

DiffuseLight::DiffuseLight(TexturePtr& a)
    : m_Emit(a)
{
}

DiffuseLight::DiffuseLight(Color& color)
    : m_Emit(std::make_shared<SolidColorTexture>(color))
{
}

bool DiffuseLight::Scatter(const Ray& ray, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const
{
    return false;
}

Material::Color DiffuseLight::Emitted(Coord& coord, const Point3& p) const
{
    return m_Emit->ColorValue(coord, p);
}
