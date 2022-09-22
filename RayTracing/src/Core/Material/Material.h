#pragma once

#include "Core/BaseObject.h"
#include "MaterialBase.h"

enum MaterialType : uint32_t {
	UNKNOWN_MATERIAL = 0x00,

	DIELECTRIC       = 0x02,
	DIFFUSE_LIGHT,
	ISOTROPIC,
	LAMBERTIAN,
	METAL,
	SHINY_METAL,

};

class Material : private MaterialBase<MaterialType>, public virtual BaseObject<Material>
{
public:

	/// <summary>
	/// 
	/// </summary>
	/// <param name="material"></param>
	virtual void AddMaterial(std::shared_ptr<Material> material) 
	{
		m_Material = material;
	}

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Utils::Math::Color4& attenuation, Ray& scattered) const override = 0;

	virtual Utils::Math::Color4 Emitted(Utils::Math::Coord& coord, const Utils::Math::Point3& p) const override
	{
		return Utils::Math::Color4(0.0f);
	}

	virtual inline MaterialType GetType() const override { return MaterialType::UNKNOWN_MATERIAL; }

	template<typename T>
	inline T* GetInstance()
	{
		static_assert(std::is_base_of<Material, T>(), "This is not a subclass of Material");
		return dynamic_cast<T*>(this);
	}

	inline Material* GetInstance() { return this; }

protected:

	std::shared_ptr<Material> m_Material = nullptr;


};