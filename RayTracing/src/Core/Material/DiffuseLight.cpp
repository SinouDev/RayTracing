#include "DiffuseLight.h"

#include "Core/Texture/SolidColorTexture.h"

using Utils::Math::Color3;
using Utils::Math::Color4;
using Utils::Math::Coord;
using Utils::Math::Point3;

DiffuseLight::DiffuseLight(TexturePtr& a)
    : m_Emit(a)
{
}

DiffuseLight::DiffuseLight(Color3& color)
    : m_Emit(std::make_shared<SolidColorTexture>(color))
{
}

bool DiffuseLight::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{
    return false;
}

Color3 DiffuseLight::Emitted(Coord& coord, const Point3& p) const
{
    return m_Emit->ColorValue(coord, p);
}
