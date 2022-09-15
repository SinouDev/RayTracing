#pragma once

#include "glm/glm.hpp"

#include "Utils/Math.h"

class Texture
{
protected:

public:

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const = 0;

};