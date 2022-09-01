#pragma once

#include "../Core/Ray.h"
#include "HittableObject.h"

class Sphere : public HittableObject
{
public:

	Sphere() = default;
	Sphere(point3& center, float r, std::shared_ptr<Material>& material);

	virtual bool Hit(Ray& ray, double min, double max, HitRecord& hitRecord) const override;

	void SetCenter(const point3& center);

	inline point3& GetCenter() { return m_Center; }
	inline float* GetRadius() { return &m_Radius; }

private:

	point3 m_Center;
	float m_Radius;
	std::shared_ptr<Material> m_Material;
};

