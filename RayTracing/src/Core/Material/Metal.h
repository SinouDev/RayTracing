#pragma once

#include "Material.h"

class Metal : public Material
{
public:

	Metal(Color& color, float fuzz);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const override;

	inline float* GetFuzz() { return &m_Fuzz; }

protected:
	Color m_Albedo;
	float m_Fuzz;

};

class ShinyMetal : public Metal
{

public:
	ShinyMetal(Color& color);

};