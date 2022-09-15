#pragma once

#include "Material.h"
#include "Core/Texture/Texture.h"

#include <memory>

class DiffuseLight : public Material 
{
	using TexturePtr = std::shared_ptr<Texture>;

public:

	DiffuseLight(TexturePtr& a);
	DiffuseLight(Utils::Math::Color3& color);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override;
	virtual Utils::Math::Color3 Emitted(Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override;

	inline TexturePtr& GetEmit() { return m_Emit; }

private:

	TexturePtr m_Emit;

};