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

	virtual inline TextureType GetType() const override { return TextureType::CHECKER_TEXTURE; }

	inline std::shared_ptr<Texture>& GetEven() { return m_Even; }
	inline std::shared_ptr<Texture>& GetOdd() { return m_Odd; }
	inline uint32_t& GetSize() { return m_Size; }

	virtual inline std::shared_ptr<Texture> Clone() const override
	{
		return nullptr;
	}

private:

	std::shared_ptr<Texture> m_Even;
	std::shared_ptr<Texture> m_Odd;
	uint32_t m_Size = 5;
};