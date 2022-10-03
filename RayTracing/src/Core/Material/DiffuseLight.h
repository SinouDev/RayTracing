#pragma once

#include "Material.h"
#include "Core/Texture/Texture.h"

#include <memory>

class DiffuseLight : public Material 
{
	using TexturePtr = std::shared_ptr<Texture>;

public:

	DiffuseLight(TexturePtr& a, float brightness = 1.0f);
	DiffuseLight(Utils::Math::Color3& color, float brightness = 1.0f);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override;
	virtual Utils::Math::Color4 Emitted(Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override;

	virtual inline MaterialType GetType() const override { return MaterialType::DIFFUSE_LIGHT; }

	inline void SetBrightness(float brightness) { brightness = Utils::Math::Max(brightness, 0.0001f); }
	inline TexturePtr& GetEmit() { return m_Emit; }
	inline float& GetBrightness() { return m_Brightness; }

	virtual inline std::shared_ptr<Material> Clone() const override
	{
		return nullptr;
	}

private:

	TexturePtr m_Emit;
	float m_Brightness = 1.0f;

};