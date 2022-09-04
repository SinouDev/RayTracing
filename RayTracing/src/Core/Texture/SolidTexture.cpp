#include "SolidTexture.h"

#include "Core/Utils.h"

SolidTexture::SolidTexture()
{
}

SolidTexture::SolidTexture(const Color& c)
	: m_ColorValue(c)
{
}

SolidTexture::SolidTexture(uint8_t r, uint8_t g, uint8_t b)
	: m_ColorValue(Utils::Color::RGBAtoVec3(r, g, b))
{
}

Texture::Color SolidTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	return m_ColorValue;
}
