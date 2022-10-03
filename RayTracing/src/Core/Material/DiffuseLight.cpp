#include "DiffuseLight.h"

#include "Core/Texture/SolidColorTexture.h"

using Utils::Math::Color3;
using Utils::Math::Color4;
using Utils::Math::Coord;
using Utils::Math::Point3;

DiffuseLight::DiffuseLight(TexturePtr& a, float brightness)
    : m_Emit(a), m_Brightness(brightness)
{
}

DiffuseLight::DiffuseLight(Color3& color, float brightness)
    : m_Emit(std::make_shared<SolidColorTexture>(color)), m_Brightness(brightness)
{
}

bool DiffuseLight::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{
    return false;
}

Color4 DiffuseLight::Emitted(Coord& coord, const Point3& p) const
{
    return Color4(Color3(m_Emit->ColorValue(coord, p)), m_Brightness);
}
