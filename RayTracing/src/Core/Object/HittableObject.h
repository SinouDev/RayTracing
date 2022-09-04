#pragma once

#include "Core/Ray.h"
#include "glm/glm.hpp"

#include <memory>

class Material;
class AABB;

struct HitRecord {

	using Point3 = glm::vec3;
	using Vec3 = glm::vec3;
	using Color = glm::vec3;
	using Coord = glm::vec2;

	Point3 point;
	Vec3 normal;
	std::shared_ptr<Material> material_ptr;
	float t;
	Coord coord;
	bool front_face;

	inline void set_face_normal(const Ray& ray, Vec3 outward_normal)
	{
		front_face = glm::dot(ray.GetDirection(), outward_normal) < 0.0f;
		normal = front_face ? outward_normal : -outward_normal;	
	}

};

class HittableObject
{
public:

	using Point3 = glm::vec3;
	using Vec3 = glm::vec3;
	using Color = glm::vec3;
	using Coord = glm::vec2;

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const = 0;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const = 0;
};