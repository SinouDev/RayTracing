#pragma once

#include "Utils/Math.h"
#include "Core/Ray.h"

struct HitRecord;

template<typename T>
class MaterialBase
{
public:

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ray"></param>
	/// <param name="hitRecord"></param>
	/// <param name="attenuation"></param>
	/// <param name="scattered"></param>
	/// <returns></returns>
	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const = 0;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="coord"></param>
	/// <param name="p"></param>
	/// <returns></returns>
	virtual Utils::Math::Color4 Emitted(Utils::Math::Coord& coord, const Utils::Math::Point3& p) const = 0;

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	virtual inline T GetType() const = 0;

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	virtual bool GetLightEmit() const { return false; }

};