#pragma once

#include "Utils/Math.h"

#include <memory>

template<typename T>
class TextureBase
{
protected:

public:

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const = 0;

	virtual inline T GetType() const = 0;

};