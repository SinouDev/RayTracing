#include "CheckerTexture.h"

#include "SolidColorTexture.h"

CheckerTexture::CheckerTexture(std::shared_ptr<Texture>& even, std::shared_ptr<Texture>& odd)
	:m_Even(even), m_Odd(odd)
{
}

CheckerTexture::CheckerTexture(Color& even, Color& odd)
	: m_Even(std::make_shared<SolidColorTexture>(even)), m_Odd(std::make_shared<SolidColorTexture>(odd))
{
}

Texture::Color CheckerTexture::ColorValue(const Coord& coord, const Point3& p) const
{
	float sines = glm::sin(m_Size * p.x) * glm::sin(m_Size * p.y) * glm::sin(m_Size * p.z);

	if (sines < 0.0f)
	{
		return m_Odd->ColorValue(coord, p);
	}
	return m_Even->ColorValue(coord, p);
}
