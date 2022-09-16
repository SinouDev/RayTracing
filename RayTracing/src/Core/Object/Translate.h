#pragma once

#include "HittableObject.h"

#include <memory>

class Translate : public HittableObject
{
public:
	Translate(std::shared_ptr<HittableObject>& object, const Utils::Math::Vec3& displacement);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return TRANSLATE; }

	inline Utils::Math::Vec3& GetTranslatePosition() { return m_Offset; }
	inline std::shared_ptr<HittableObject>& GetObject() { return m_Object; }

private:
	std::shared_ptr<HittableObject> m_Object;
	Utils::Math::Vec3 m_Offset;
};