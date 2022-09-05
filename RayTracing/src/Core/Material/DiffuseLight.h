#pragma once

#include "Material.h"
#include "Core/Texture/Texture.h"

#include <memory>

class DiffuseLight : public Material 
{
	using TexturePtr = std::shared_ptr<Texture>;

public:

	DiffuseLight(TexturePtr& a);
	DiffuseLight(Color& color);

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Color& attenuation, Ray& scattered) const override;
	virtual Color Emitted(Coord& coord, const Point3& p) const override;

	inline TexturePtr& GetEmit() { return m_Emit; }

private:

	TexturePtr m_Emit;

};