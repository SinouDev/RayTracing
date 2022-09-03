#pragma once

#include "Core/Ray.h"
#include "HittableObject.h"

class Sphere : public HittableObject
{
public:

	Sphere() = default;
	Sphere(point3& center, float r, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	void SetCenter(const point3& center);

	inline point3& GetCenter() { return m_Center; }
	inline float* GetRadius() { return &m_Radius; }

private:

	point3 m_Center;
	float m_Radius;
	std::shared_ptr<Material> m_Material;
};

