#pragma once

#include "Core/Ray.h"

#include "Utils/Math.h"

#include <memory>

struct HitRecord;

class Material
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
	virtual Utils::Math::Color3 Emitted(Utils::Math::Coord& coord, const Utils::Math::Point3& p) const
	{
		return Utils::Math::Color3(0.0f);
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="material"></param>
	virtual void AddMaterial(std::shared_ptr<Material> material) 
	{
		m_Material = material;
	}


protected:

	std::shared_ptr<Material> m_Material = nullptr;


};