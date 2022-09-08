#pragma once

#include "Core/Ray.h"

#include "glm/glm.hpp"
#include <memory>

struct HitRecord;

class Material
{
public:

	using Color  = glm::vec3;
	using Color4 = glm::vec4;
	using Point3 = glm::vec3;
	using Vec3   = glm::vec3;
	using Coord  = glm::vec2;

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, Color4& attenuation, Ray& scattered) const = 0;
	virtual Color Emitted(Coord& coord, const Point3& p) const
	{
		return Color(0.0f);
	}
	virtual void AddMaterial(std::shared_ptr<Material> material) 
	{
		m_Material = material;
	}


protected:

	std::shared_ptr<Material> m_Material = nullptr;

	static Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat)
	{
		float cos_theta = glm::min(glm::dot(-uv, n), 1.0f);
		Vec3 r_out_perp = (uv + cos_theta * n) * etai_over_etat;
		Vec3 r_out_paralle = -glm::sqrt(glm::abs(1.0f - glm::dot(r_out_perp, r_out_perp))) * n;
		return r_out_perp + r_out_paralle;
	}

	static Vec3 reflect(const Vec3& v, const Vec3& n)
	{
		return v - 2.0f * glm::dot(v, n) * n;
	}


};