#pragma once

#include "Material.h"

#include "Core/Texture/Texture.h"

#include <memory>

class Metal : public Material
{
public:

	Metal(Color& color, float fuzz);
	Metal(std::shared_ptr<Texture>& texture, float fuzz);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const override;

	inline float* GetFuzz() { return &m_Fuzz; }

protected:
	std::shared_ptr<Texture> m_Albedo;
	float m_Fuzz;

};

class ShinyMetal : public Metal
{

public:
	ShinyMetal(Color& color);
	ShinyMetal(std::shared_ptr<Texture>& texture);

};