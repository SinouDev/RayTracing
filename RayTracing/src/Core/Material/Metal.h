#pragma once

#include "Material.h"

#include "Core/Texture/Texture.h"

#include <memory>

class Metal : public Material
{
public:

	Metal(Utils::Math::Color3& color, float fuzz);
	Metal(std::shared_ptr<Texture>& texture, float fuzz);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override;

	virtual inline MaterialType GetType() const override { return MaterialType::METAL; }

	inline float* GetFuzz() { return &m_Fuzz; }

	inline std::shared_ptr<Texture>& GetAlbedo() { return m_Albedo; }

	virtual inline std::shared_ptr<Material> Clone() const override
	{
		return nullptr;
	}

protected:
	std::shared_ptr<Texture> m_Albedo;
	float m_Fuzz;

};

class ShinyMetal : public Metal
{

public:
	ShinyMetal(Utils::Math::Color3& color);
	ShinyMetal(std::shared_ptr<Texture>& texture);

	virtual inline MaterialType GetType() const override { return MaterialType::SHINY_METAL; }

};