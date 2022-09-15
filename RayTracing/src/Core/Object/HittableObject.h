#pragma once

#include "Core/Ray.h"

#include "Utils/Math.h"

#include <memory>

class Material;
class AABB;

struct HitRecord {

	Utils::Math::Point3 point{ 0.0f };
	Utils::Math::Vec3 normal{ 0.0f };
	Utils::Math::Coord coord{ 0.0f };
	std::shared_ptr<Material> material_ptr = nullptr;
	float t = 0.0f;
	bool front_face = false;

	inline void set_face_normal(const Ray& ray, Utils::Math::Vec3 outward_normal)
	{
		front_face = Utils::Math::Dot(ray.GetDirection(), outward_normal) < 0.0f;
		normal = front_face ? outward_normal : -outward_normal;	
	}

};

class HittableObject
{
public:

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const = 0;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const = 0;
};