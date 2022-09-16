#pragma once

#include "HittableObject.h"

class XyRect : public HittableObject
{
public:

	XyRect();
	XyRect(const Utils::Math::Mat2x2& pos, float k, const std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return XY_RECT; }

	inline Utils::Math::Mat2x2& GetPositions() { return m_Pos; }

private:
	Utils::Math::Mat2x2 m_Pos;
	float m_K;
	std::shared_ptr<Material> m_Material;
};