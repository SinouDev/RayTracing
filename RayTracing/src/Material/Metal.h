#pragma once

#include "Material.h"

class Metal : public Material
{
public:

	Metal(color& color, float fuzz);

	virtual bool Scatter(Ray& ray, HitRecord& hitRecord, color& attenuation, Ray& scattered) const override;

	inline float* GetFuzz() { return &m_Fuzz; }

protected:
	color m_Albedo;
	float m_Fuzz;

};

class ShinyMetal : public Metal
{

public:
	ShinyMetal(color& color);

};