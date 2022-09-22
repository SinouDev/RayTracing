#pragma once

#include "Texture.h"

class Texture2D : public Texture
{
private:
	const static int32_t s_TextureChannels = 4;

public:
	Texture2D();
	Texture2D(const char* file_name);
	~Texture2D();

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override;

	inline const char* GetFileName() const { return m_FileName; }

	virtual inline TextureType GetType() const override { return TextureType::TEXTURE_2D; }

	virtual inline std::shared_ptr<Texture> Clone() const override
	{
		return nullptr;
	}

private:

	uint32_t* m_Data;
	uint32_t m_Width, m_Height;
	uint32_t m_Channels;
	const char* m_FileName;
};