#pragma once

#include "Material.h"

class Dielectric : public Material 
{
public:

	Dielectric(float index_of_refraction);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override;

	inline float& GetIndexOfRefraction() { return m_IndexOfRefraction; }

	virtual inline MaterialType GetType() const override { return MaterialType::DIELECTRIC; }

	virtual inline std::shared_ptr<Material> Clone() const override
	{
		return nullptr;
	}

private:

	float m_IndexOfRefraction;

};