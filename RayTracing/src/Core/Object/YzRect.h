#pragma once

#include "HittableObject.h"

class YzRect : public HittableObject
{
public:

	YzRect();
	YzRect(const Utils::Math::Mat2x2& pos, float k, const std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

private:
	Utils::Math::Mat2x2 m_Pos;
	float m_K;
	std::shared_ptr<Material> m_Material;
};