#pragma once

#include "HittableObject.h"

class Material;

class MovingSphere : public HittableObject
{
public:
	MovingSphere();
	MovingSphere(Point3& cen0, Point3& cen1, float _time0, float _time1, float radius, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	Point3 GetCenter(float time) const;

private:

	Point3 m_Center0, m_Center1;
	float m_Time0, m_Time1;
	float m_Radius;
	std::shared_ptr<Material> m_Material;

};