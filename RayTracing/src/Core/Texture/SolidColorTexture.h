#pragma once

#include "Texture.h"

class SolidColorTexture : public Texture
{
public:
	SolidColorTexture();
	SolidColorTexture(const Utils::Math::Color3& color);
	SolidColorTexture(const Utils::Math::Color4& color);
	SolidColorTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override;

	inline Utils::Math::Color4& GetColor() { return m_ColorValue; }

	virtual inline TextureType GetType() const override { return TextureType::SOLID_COLOR_TEXTURE; }

	virtual inline std::shared_ptr<Texture> Clone() const override
	{
		return std::shared_ptr<SolidColorTexture>(new SolidColorTexture(m_ColorValue));
	}

private:

	Utils::Math::Color4 m_ColorValue{ 0.0f };

};