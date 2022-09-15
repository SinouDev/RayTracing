#pragma once

#include "Texture.h"

#include "Core/Perlin.h"

class NoiseTexture : public Texture
{
public:

	NoiseTexture();
	NoiseTexture(float scale);

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override;

private:

	Perlin m_Noise;
	float m_Scale = 1.0f;
};