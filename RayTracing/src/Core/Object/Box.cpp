#include "Box.h"

#include "XyRect.h"
#include "XzRect.h"
#include "YzRect.h"

using Utils::Math::Point3;
using Utils::Math::Point2;
using Utils::Math::Mat2x2;

Box::Box()
{
}

Box::Box(const Point3& point1, const Point3& point2, std::shared_ptr<Material>& material)
{
	m_BoxMin = point1;
	m_BoxMax = point2;

	m_Sides.Add(std::make_shared<XyRect>(Mat2x2{ Point2(point1),             Point2(point2) },             point2.z, material));
	m_Sides.Add(std::make_shared<XyRect>(Mat2x2{ Point2(point1),             Point2(point2) },             point1.z, material));

	m_Sides.Add(std::make_shared<XzRect>(Mat2x2{ Point2(point1.x, point1.z), Point2(point2.x, point2.z) }, point2.y, material));
	m_Sides.Add(std::make_shared<XzRect>(Mat2x2{ Point2(point1.x, point1.z), Point2(point2.x, point2.z) }, point1.y, material));

	m_Sides.Add(std::make_shared<YzRect>(Mat2x2{ Point2(point1.y, point1.z), Point2(point2.y, point2.z) }, point2.x, material));
	m_Sides.Add(std::make_shared<YzRect>(Mat2x2{ Point2(point1.y, point1.z), Point2(point2.y, point2.z) }, point1.x, material));
}

bool Box::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	return m_Sides.Hit(ray, min, max, hitRecord);
}

bool Box::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	output_box = AABB(m_BoxMin, m_BoxMax);
	return true;
}
