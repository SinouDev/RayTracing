#pragma once

#include "Core/Ray.h"

#include "glm/glm.hpp"

struct HitRecord;

class Material
{
public:

	using color  = glm::vec3;
	using point3 = glm::vec3;
	using vec3   = glm::vec3;

	virtual bool Scatter(const Ray& ray, const HitRecord& hitRecord, color& attenuation, Ray& scattered) const = 0;


protected:

	static vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat)
	{
		float cos_theta = glm::min(glm::dot(-uv, n), 1.0f);
		vec3 r_out_perp = (uv + cos_theta * n) * etai_over_etat;
		vec3 r_out_paralle = -glm::sqrt(glm::abs(1.0f - glm::dot(r_out_perp, r_out_perp))) * n;
		return r_out_perp + r_out_paralle;
	}

	static vec3 reflect(const vec3& v, const vec3& n)
	{
		return v - 2.0f * glm::dot(v, n) * n;
	}


};