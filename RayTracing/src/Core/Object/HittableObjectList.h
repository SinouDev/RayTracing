#pragma once

#include "HittableObject.h"

#include <memory>
#include <vector>

class HittableObjectList : public HittableObject
{

public:

	HittableObjectList() = default;
	HittableObjectList(std::shared_ptr<HittableObject>& object);

	void Clear();
	void Add(const std::shared_ptr<HittableObject>& object);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

private:

	std::vector<std::shared_ptr<HittableObject>> m_Objects;
};