#pragma once

#include "HittableObject.h"

#include <memory>

class Translate : public HittableObject
{
public:
	Translate(std::shared_ptr<HittableObject>& object, const Vec3& displacement);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

private:
	std::shared_ptr<HittableObject> m_Object;
	Vec3 m_Offset;
};