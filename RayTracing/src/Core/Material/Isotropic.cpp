#include "Isotropic.h"

#include "Utils/Random.h"
#include "Core/Object/HittableObject.h"
#include "Core/Texture/SolidColorTexture.h"

using Utils::Math::Color3;
using Utils::Math::Color4;

Isotropic::Isotropic(Color3& color)
	: m_Albedo(std::make_shared<SolidColorTexture>(color))
{
}

Isotropic::Isotropic(std::shared_ptr<Texture>& texture)
	: m_Albedo(texture)
{
}

bool Isotropic::Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const
{
	scattered = Ray(hitRecord.point, Utils::Random::RandomInUnitSphere(), ray.GetTime());
	attenuation = m_Albedo->ColorValue(hitRecord.coord, hitRecord.point);
	return true;
}
