#pragma once

#include "../Core/Ray.h"
#include "glm/glm.hpp"

#include <memory>

class Material;

struct HitRecord {

	using point3 = glm::vec3;
	using vec3 = glm::vec3;
	using color = glm::vec3;

	point3 point;
	vec3 normal;
	std::shared_ptr<Material> material_ptr;
	float t;
	bool front_face;

	inline void set_face_normal(Ray& ray, vec3 outward_normal)
	{
		front_face = glm::dot(ray.GetDirection(), outward_normal) < 0.0f;
		normal = front_face ? outward_normal : -outward_normal;	
	}

};

class HittableObject
{
public:

	using point3 = glm::vec3;
	using vec3 = glm::vec3;
	using color = glm::vec3;

	virtual bool Hit(Ray& ray, double min, double max, HitRecord& hitRecord) const = 0;
};