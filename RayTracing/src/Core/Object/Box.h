#pragma once

#include "HittableObject.h"
#include "HittableObjectList.h"

class Box : public HittableObject
{
public:

	Box();
	Box(const Point3& point1, const Point3& point2, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

private:

	Point3 m_BoxMin;
	Point3 m_BoxMax;
	HittableObjectList m_Sides;

};