#pragma once

#include "HittableObject.h"

class Material;

class MovingSphere : public HittableObject
{
public:
	MovingSphere();
	MovingSphere(Utils::Math::Point3& cen0, Utils::Math::Point3& cen1, float _time0, float _time1, float radius, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return MOVING_SPHERE; }

	Utils::Math::Point3 GetCenter(float time) const;

	inline Utils::Math::Point3& GetCenter0() { return m_Center0; }
	inline Utils::Math::Point3& GetCenter1() { return m_Center1; }

private:

	Utils::Math::Point3 m_Center0, m_Center1;
	float m_Time0, m_Time1;
	float m_Radius;
	std::shared_ptr<Material> m_Material;

};