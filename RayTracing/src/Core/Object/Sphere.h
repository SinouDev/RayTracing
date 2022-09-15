#pragma once

#include "Core/Ray.h"
#include "HittableObject.h"

class Sphere : public HittableObject
{
public:

	Sphere() = default;
	Sphere(Utils::Math::Point3& center, float r, std::shared_ptr<Material>& material);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	void SetCenter(const Utils::Math::Point3& center);

	inline Utils::Math::Point3& GetCenter() { return m_Center; }
	inline float* GetRadius() { return &m_Radius; }

private:

	static void GetSphereCoord(const Utils::Math::Point3& p, Utils::Math::Coord& coord);

private:

	Utils::Math::Point3 m_Center{ 0.0f };
	float m_Radius = 0.0f;
	std::shared_ptr<Material> m_Material = nullptr;
};

