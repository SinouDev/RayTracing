#pragma once

#include "Material.h"

class Dielectric : public Material 
{
public:

	Dielectric(float index_of_refraction);

	virtual bool Scatter(Ray& ray, HitRecord& hitRecord, color& attenuation, Ray& scattered) const override;

	inline float* GetIndexOfRefraction() { return &m_IndexOfRefraction; }

protected:

	static float reflectness(float cosine, float ref_index);

private:

	float m_IndexOfRefraction;

};