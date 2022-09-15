#pragma once

#include "Texture.h"

#include <memory>

class CheckerTexture : public Texture
{
public:

	CheckerTexture(std::shared_ptr<Texture>& even, std::shared_ptr<Texture>& odd);
	CheckerTexture(Utils::Math::Color3& even, Utils::Math::Color3& odd);
	CheckerTexture(Utils::Math::Color4& even, Utils::Math::Color4& odd);

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override;

private:

	std::shared_ptr<Texture> m_Even;
	std::shared_ptr<Texture> m_Odd;
	uint32_t m_Size = 5;
};