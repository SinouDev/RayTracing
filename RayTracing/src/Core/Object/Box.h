#pragma once

#include "HittableObject.h"
#include "HittableObjectList.h"

class Box : public HittableObject
{
public:

	Box();
	Box(const Utils::Math::Point3& point1, const Utils::Math::Point3& point2, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return BOX; }

	inline HittableObjectList* GetSides() { return &m_Sides; }

private:

	Utils::Math::Point3 m_BoxMin {0.0f};
	Utils::Math::Point3 m_BoxMax {0.0f};
	HittableObjectList m_Sides;

};