#pragma once

#include "glm/glm.hpp"

class Texture
{
protected:
	using Color4 = glm::vec4;
	using Color = glm::vec3;
	using Coord = glm::vec2;
	using Point3 = glm::vec3;

public:

	virtual Color4 ColorValue(const Coord& coord, const Point3& p) const = 0;

};