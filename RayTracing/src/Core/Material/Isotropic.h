#pragma once

#include "Material.h"

#include "Core/Texture/Texture.h"

#include <memory>

class Isotropic : public Material
{
public:

	Isotropic(Color& color);
	Isotropic(std::shared_ptr<Texture>& texture);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const override;

private:

	std::shared_ptr<Texture> m_Albedo;
};