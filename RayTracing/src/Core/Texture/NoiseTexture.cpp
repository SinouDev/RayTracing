#include "NoiseTexture.h"

NoiseTexture::NoiseTexture()
{
}

NoiseTexture::NoiseTexture(float scale)
	: m_Scale(scale)
{
}

Texture::Color NoiseTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	//return Color(1.0f) * 0.5f * (1.0f + m_Noise.Noise(m_Scale * p));
	//return Color(1.0f) * m_Noise.Turbulence(m_Scale * p);
	return Color(1.0f) * 0.5f * (1.0f + glm::sin(m_Scale * p.z + 10.0f * m_Noise.Turbulence(p)));
}
