#pragma once

#include "Texture.h"

#include <memory>

class CheckerTexture : public Texture
{
public:

	CheckerTexture(std::shared_ptr<Texture>& even, std::shared_ptr<Texture>& odd);
	CheckerTexture(Color& even, Color& odd);
	CheckerTexture(Color4& even, Color4& odd);

	virtual Color4 ColorValue(const Coord& coord, const Point3& p) const override;

private:

	std::shared_ptr<Texture> m_Even;
	std::shared_ptr<Texture> m_Odd;
	uint32_t m_Size = 5;
};