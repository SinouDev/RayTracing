#pragma once

#include "Material.h"

#include "Core/Texture/Texture.h"

#include <memory>

class Isotropic : public Material
{

public:

	Isotropic(Utils::Math::Color3& color);
	Isotropic(std::shared_ptr<Texture>& texture);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override;

	virtual inline MaterialType GetType() const override { return MaterialType::ISOTROPIC; }

	inline std::shared_ptr<Texture>& GetAlbedo() { return m_Albedo; }

	virtual inline std::shared_ptr<Material> Clone() const override
	{
		return nullptr;
	}

private:

	std::shared_ptr<Texture> m_Albedo;
};