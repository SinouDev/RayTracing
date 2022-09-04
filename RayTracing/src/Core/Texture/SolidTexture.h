#pragma once

#include "Texture.h"

class SolidTexture : public Texture
{
public:
	SolidTexture();
	SolidTexture(const Color&);
	SolidTexture(uint8_t r, uint8_t g, uint8_t b);

	virtual Color ColorValue(const Coord& coord, const Point3& p) const override;

private:

	Color m_ColorValue;

};