#include "SolidColorTexture.h"

#include "Core/Utils.h"

SolidColorTexture::SolidColorTexture()
{
}

SolidColorTexture::SolidColorTexture(const Color& color)
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

Texture::Color4 SolidColorTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	return m_ColorValue;
}
