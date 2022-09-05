#pragma once

#include "HittableObject.h"
#include "Core/AABB.h"

#include <memory>
#include <vector>

class HittableObjectList;

class BVHnode : public HittableObject
{
private:

	using HittablePtr = std::shared_ptr<HittableObject>;
	using HittableList = std::vector<HittablePtr>;

public:

	BVHnode();
	BVHnode(const HittableObjectList& list, float _time0, float _time1);
	BVHnode(const HittableList& srcObjects, size_t start, size_t end, float _time0, float _time1);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	

private:

	static inline bool BoxCompare(const HittablePtr& a, const HittablePtr& b, int32_t axis);

	friend bool box_compare_x(const HittablePtr& a, const HittablePtr& b);
	friend bool box_compare_y(const HittablePtr& a, const HittablePtr& b);
	friend bool box_compare_z(const HittablePtr& a, const HittablePtr& b);

private:

	HittablePtr m_Left;
	HittablePtr m_Right;
	AABB m_Box;

};