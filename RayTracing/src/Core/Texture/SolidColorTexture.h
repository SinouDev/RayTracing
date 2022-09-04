#pragma once

#include "Texture.h"

class SolidColorTexture : public Texture
{
public:
	SolidColorTexture();
	SolidColorTexture(const Color&);
	SolidColorTexture(uint8_t r, uint8_t g, uint8_t b);

	virtual Color ColorValue(const Coord& coord, const Point3& p) const override;

private:

	Color m_ColorValue;

};