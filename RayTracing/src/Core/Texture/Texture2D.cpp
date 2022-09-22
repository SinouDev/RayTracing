#include "Texture2D.h"

#include "Utils/Color.h"

#include "stb_image.h"

#include <iostream>

using Utils::Math::Color4;
using Utils::Math::Coord;
using Utils::Math::Point3;

Texture2D::Texture2D()
	: m_Data(nullptr), m_Width(0), m_Height(0), m_Channels(0), m_FileName(nullptr)
{
}

Texture2D::Texture2D(const char* file_name)
{
	m_FileName = file_name;
	m_Data = (uint32_t*) stbi_load(file_name, (int32_t*) & m_Width, (int32_t*) & m_Height, (int32_t*) & m_Channels, s_TextureChannels);

	if (!m_Data)
		std::cerr << "ERROR: the image file '" << file_name << "' couldn't be loaded.\n";
}

Texture2D::~Texture2D()
{
	delete[] m_Data;
}

Color4 Texture2D::ColorValue(const Coord& coord, const Point3& p) const
{
	Color4 color(0.0f, 1.0f, 1.0f, 1.0f);
	if(!m_Data)
		return color;

	Coord c;

	c = Utils::Math::Clamp(coord, 0.0f, 1.0f);
	c.t = 1.0f - c.t;

	uint32_t x = static_cast<uint32_t>(c.s * m_Width);
	uint32_t y = static_cast<uint32_t>(c.t * m_Height);

	x = x < m_Width ? x : m_Width;
	y = y < m_Height ? y : m_Height;

	size_t px = (size_t)x + y * m_Width;

	//if (px < (m_Width * m_Height * s_TextureChannels) / sizeof(uint32_t) )
	
	Utils::Color::RGBAtoVec4(color, m_Data[Utils::Math::Min(px, (size_t)m_Width * m_Height - 1)]); // it causes memmory access violation sometimes

	return color;

}