#include "SolidColorTexture.h"

#include "Core/Utils.h"

SolidColorTexture::SolidColorTexture()
{
}

SolidColorTexture::SolidColorTexture(const Color& c)
	: m_ColorValue(c)
{
}

SolidColorTexture::SolidColorTexture(uint8_t r, uint8_t g, uint8_t b)
	: m_ColorValue(Utils::Color::RGBAtoVec3(r, g, b))
{
}

Texture::Color SolidColorTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	return m_ColorValue;
}
