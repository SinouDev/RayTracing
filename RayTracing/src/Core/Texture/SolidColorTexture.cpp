#include "SolidColorTexture.h"

#include "Utils/Color.h"

using Utils::Math::Color3;
using Utils::Math::Color4;
using Utils::Math::Coord;
using Utils::Math::Point3;

SolidColorTexture::SolidColorTexture()
{
}

SolidColorTexture::SolidColorTexture(const Color3& color)
	: SolidColorTexture(Color4(color, 1.0f))
{
}

SolidColorTexture::SolidColorTexture(const Color4& color)
	: m_ColorValue(color)
{
}

SolidColorTexture::SolidColorTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
	: m_ColorValue(Utils::Color::RGBAtoVec4(r, g, b, a))
{
}

Color4 SolidColorTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	return m_ColorValue;
}
