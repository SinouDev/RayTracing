#pragma once

#include "Material.h"
#include "Core/Texture/Texture.h"

#include <memory>

class Lambertian : public Material
{
public:

	Lambertian(Color& color);
	Lambertian(std::shared_ptr<Texture>& texture);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const override;

private:
	static bool near_zero(Material::Vec3& vector);

private:
	std::shared_ptr<Texture> m_Albedo;

};