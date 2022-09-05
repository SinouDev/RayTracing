#pragma once

#include "Texture.h"

class SolidColorTexture : public Texture
{
public:
	SolidColorTexture();
	SolidColorTexture(const Color&);
	SolidColorTexture(uint8_t r, uint8_t g, uint8_t b);

	virtual Color ColorValue(const Coord& coord, const Point3& p) const override;

	inline Color& GetColor() { return m_ColorValue; }

private:

	Color m_ColorValue;

};