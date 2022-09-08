#pragma once

#include "Texture.h"

class SolidColorTexture : public Texture
{
public:
	SolidColorTexture();
	SolidColorTexture(const Color& color);
	SolidColorTexture(const Color4& color);
	SolidColorTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

	virtual Color4 ColorValue(const Coord& coord, const Point3& p) const override;

	inline Color4& GetColor() { return m_ColorValue; }

private:

	Color4 m_ColorValue;

};