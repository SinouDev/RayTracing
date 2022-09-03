#pragma once

#include "Material.h"

class Lambertian : public Material
{
public:

	Lambertian(color& color);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, color& attenuation, Ray& scattered) const override;

private:
	static bool near_zero(Material::vec3& vector);

private:
	color m_Albedo;

};