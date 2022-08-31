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

	virtual bool Hit(Ray& ray, double min, double max, HitRecord& hitRecord) const override;

private:

	std::vector<std::shared_ptr<HittableObject>> m_Objects;
};