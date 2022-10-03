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

	virtual inline MaterialType GetType() const override { return MaterialType::LAMBERTIAN; }

	inline std::shared_ptr<Texture>& GetAlbedo() { return m_Albedo; }

	virtual inline std::shared_ptr<Material> Clone() const override
	{
		Lambertian* copy = new Lambertian(m_Albedo->Clone());
		return std::shared_ptr<Lambertian>(copy);
	}

private:

private:
	std::shared_ptr<Texture> m_Albedo;

};