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

	virtual inline HittableObjectTypes GetType() const override { return HittableObjectTypes::BOX; }

	inline HittableObjectList* GetSides() { return &m_Sides; }

	virtual inline std::shared_ptr<HittableObject> Clone() const override
	{
		return nullptr;
	}

	virtual inline Utils::Math::Vec3& GetObjectTranslate() override { return m_ObjectTranslate; }
	virtual inline Utils::Math::Vec3& GetObjectRotate() override { return m_ObjectRotate; }
	virtual inline Utils::Math::Vec3& GetObjectScale() override { return m_ObjectScale; }

private:

	Utils::Math::Point3 m_BoxMin {0.0f};
	Utils::Math::Point3 m_BoxMax {0.0f};
	HittableObjectList m_Sides;

};