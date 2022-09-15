#include "NoiseTexture.h"

using Utils::Math::Color3;
using Utils::Math::Color4;
using Utils::Math::Point3;
using Utils::Math::Coord;

NoiseTexture::NoiseTexture()
{
}

NoiseTexture::NoiseTexture(float scale)
	: m_Scale(scale)
{
}

Color4 NoiseTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	//return Color3(1.0f) * 0.5f * (1.0f + m_Noise.Noise(m_Scale * p));
	//return Color3(1.0f) * m_Noise.Turbulence(m_Scale * p);
	return Color4(Color3(1.0f) * 0.5f * (1.0f + Utils::Math::Sin(m_Scale * p.z + 10.0f * m_Noise.Turbulence(p))), 1.0f);
}
