#include "Texture2D.h"

#include "Core/Utils.h"

#include "stb_image.h"

#include <iostream>

Texture2D::Texture2D()
	: m_Data(nullptr), m_Width(0), m_Height(0), m_Channels(0)
{
}

Texture2D::Texture2D(const char* file_name)
{
	m_Data = (uint32_t*)stbi_load(file_name, &m_Width, &m_Height, &m_Channels, s_TextureChannels);

	if (!m_Data)
		std::cerr << "ERROR: the image file '" << file_name << "' couldn't be loaded.\n";
}

Texture2D::~Texture2D()
{
	delete[] m_Data;
}

Texture::Color Texture2D::ColorValue(const Coord& coord, const Point3& p) const
{
	Color color(0.0f, 1.0f, 1.0f);
	if(!m_Data)
		return color;

	Coord c;

	c = glm::clamp(coord, 0.0f, 1.0f);
	c.t = 1.0f - c.t;

	int32_t x = static_cast<int32_t>(c.s * m_Width);
	int32_t y = static_cast<int32_t>(c.t * m_Height);

	x = x < m_Width ? x : m_Width - 1;
	y = y < m_Height ? y : m_Height - 1;

	int32_t px = x + y * m_Width;

	if (px < m_Width * m_Height * s_TextureChannels)
		Utils::Color::RGBAtoVec3(color, m_Data[px]);
	
	return color;

}
