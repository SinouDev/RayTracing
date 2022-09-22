#pragma once

#include "TextureBase.h"

#include "Core/BaseObject.h"

enum TextureType : uint32_t {
	UNKNOWN_TEXTURE      = 0x00,

	CHECKER_TEXTURE      = 0x02,
	NOISE_TEXTURE,
	SOLID_COLOR_TEXTURE,
	TEXTURE_2D,
};

class Texture : public TextureBase<TextureType>, public virtual BaseObject<Texture>
{
protected:

public:

	virtual Utils::Math::Color4 ColorValue(const Utils::Math::Coord& coord, const Utils::Math::Point3& p) const = 0;

	virtual inline TextureType GetType() const override { return TextureType::UNKNOWN_TEXTURE; }

	template<typename T>
	inline T* GetInstance()
	{
		static_assert(std::is_base_of<Texture, T>(), "This is not a subclass of Texture");
		return dynamic_cast<T*>(this);
	}

	inline Texture* GetInstance() { return this; }

};