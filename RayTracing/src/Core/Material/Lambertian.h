#pragma once

#include "Material.h"
#include "Core/Texture/Texture.h"

#include <memory>

class Lambertian : public Material
{
public:

	Lambertian(Utils::Math::Color3& color);
	Lambertian(Utils::Math::Color4& color);
	Lambertian(std::shared_ptr<Texture>& texture);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override;

private:

private:
	std::shared_ptr<Texture> m_Albedo;

};