#pragma once

#include "HittableObject.h"

class Material;

class MovingSphere : public HittableObject
{
public:
	MovingSphere();
	MovingSphere(point3& cen0, point3& cen1, float _time0, float _time1, float radius, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	point3 GetCenter(float time) const;

private:

	point3 m_Center0, m_Center1;
	float m_Time0, m_Time1;
	float m_Radius;
	std::shared_ptr<Material> m_Material;

};