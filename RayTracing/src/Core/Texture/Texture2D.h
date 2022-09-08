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

	virtual Color4 ColorValue(const Coord& coord, const Point3& p) const override;

private:

	uint32_t* m_Data;
	int32_t m_Width, m_Height;
	int32_t m_Channels;
};