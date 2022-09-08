#pragma once

#include "Texture.h"

#include "Core/Perlin.h"

class NoiseTexture : public Texture
{
public:

	NoiseTexture();
	NoiseTexture(float scale);

	virtual Color4 ColorValue(const Coord& coord, const Point3& p) const override;

private:

	Perlin m_Noise;
	float m_Scale = 1.0f;
};