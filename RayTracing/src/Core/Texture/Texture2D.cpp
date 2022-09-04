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
	m_Data = (uint32_t*)stbi_load(file_name, &m_Width, &m_Height, &m_Channels, 4);

	if (!m_Data)
		std::cerr << "ERROR: the image file '" << file_name << "' couldn't be loaded.\n";
}

Texture::Color Texture2D::ColorValue(const Coord& coord, const Point3& p) const
{
	if(!m_Data)
		return Color(0.0f, 1.0f, 1.0f);

	Coord c;

	c.s = glm::clamp(coord.s, 0.0f, 1.0f);
	c.t = 1.0f - glm::clamp(coord.t, 0.0f, 1.0f);

	int32_t x = c.s * m_Width;
	int32_t y = c.t * m_Height;

	x = x < m_Width ? x : m_Width - 1;
	y = y < m_Height ? y : m_Height - 1;

	const float color_scale = 1.0f / 255.0f;

	uint32_t px = x + y * m_Width;

	return Utils::Color::RGBAtoVec3(m_Data[px]);

}
