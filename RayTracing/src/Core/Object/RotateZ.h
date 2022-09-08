#pragma once

#include "HittableObject.h"

#include "Core/AABB.h"

#include <memory>

class RotateZ : public HittableObject
{
public:

	RotateZ(std::shared_ptr<HittableObject>& object, float angle);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

private:

	std::shared_ptr<HittableObject> m_Object;
	float m_SinTheta;
	float m_CosTheta;
	bool m_HasBox;
	AABB m_Box;

};