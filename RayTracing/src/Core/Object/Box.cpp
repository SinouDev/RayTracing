#include "Box.h"

#include "XyRect.h"
#include "XzRect.h"
#include "YzRect.h"

Box::Box()
{
}

Box::Box(const Point3& point1, const Point3& point2, std::shared_ptr<Material>& material)
{
	m_BoxMin = point1;
	m_BoxMax = point2;

	m_Sides.Add(std::make_shared<XyRect>(glm::mat2x2{ Coord(point1),             Coord(point2) },             point2.z, material));
	m_Sides.Add(std::make_shared<XyRect>(glm::mat2x2{ Coord(point1),             Coord(point2) },             point1.z, material));

	m_Sides.Add(std::make_shared<XzRect>(glm::mat2x2{ Coord(point1.x, point1.z), Coord(point2.x, point2.z) }, point2.y, material));
	m_Sides.Add(std::make_shared<XzRect>(glm::mat2x2{ Coord(point1.x, point1.z), Coord(point2.x, point2.z) }, point1.y, material));

	m_Sides.Add(std::make_shared<YzRect>(glm::mat2x2{ Coord(point1.y, point1.z), Coord(point2.y, point2.z) }, point2.x, material));
	m_Sides.Add(std::make_shared<YzRect>(glm::mat2x2{ Coord(point1.y, point1.z), Coord(point2.y, point2.z) }, point1.x, material));
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
