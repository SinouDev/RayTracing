#include "Box.h"

#include "XyRect.h"
#include "XzRect.h"
#include "YzRect.h"
#include "Translate.h"
#include "Rotate.h"

using Utils::Math::Point3;
using Utils::Math::Point2;
using Utils::Math::Mat2x2;

Box::Box()
{
	m_Name = "Box";
}

Box::Box(const Point3& point1, const Point3& point2, std::shared_ptr<Material>& material)
	: Box()
{
	m_BoxMin = point1;
	m_BoxMax = point2;

	using HittablePtr = std::shared_ptr<HittableObject>;

	HittablePtr rearFaceRect   = std::make_shared<XyRect>(Mat2x2{ Point2(point1),             Point2(point2)             }, point2.z, material);
	HittablePtr frontFaceRect  = std::make_shared<XyRect>(Mat2x2{ Point2(point1),             Point2(point2)             }, point1.z, material->Clone());

	HittablePtr topFaceRect    = std::make_shared<XzRect>(Mat2x2{ Point2(point1.x, point1.z), Point2(point2.x, point2.z) }, point2.y, material->Clone());
	HittablePtr bottomFaceRect = std::make_shared<XzRect>(Mat2x2{ Point2(point1.x, point1.z), Point2(point2.x, point2.z) }, point1.y, material->Clone());

	HittablePtr leftFaceRect   = std::make_shared<YzRect>(Mat2x2{ Point2(point1.y, point1.z), Point2(point2.y, point2.z) }, point2.x, material->Clone());
	HittablePtr rightFaceRect  = std::make_shared<YzRect>(Mat2x2{ Point2(point1.y, point1.z), Point2(point2.y, point2.z) }, point1.x, material->Clone());
	
	rearFaceRect  ->SetName("Rear Face");
	frontFaceRect ->SetName("Front Face");
	topFaceRect   ->SetName("Top Face");
	bottomFaceRect->SetName("Bottom Face");
	leftFaceRect  ->SetName("Left Face");
	rightFaceRect ->SetName("Right Face");

	HittablePtr frontFaceTranslate = std::make_shared<Translate>(frontFaceRect);

	m_Sides.Add(std::make_shared<Rotate>(frontFaceTranslate, Point3(0.0f)));
	m_Sides.Add(std::make_shared<Translate>(rearFaceRect));
	m_Sides.Add(std::make_shared<Translate>(topFaceRect));
	m_Sides.Add(std::make_shared<Translate>(bottomFaceRect));
	m_Sides.Add(std::make_shared<Translate>(leftFaceRect));
	m_Sides.Add(std::make_shared<Translate>(rightFaceRect));
}

bool Box::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	if (!m_Hittable)
		return false;

	return m_Sides.Hit(ray, min, max, hitRecord);
}

bool Box::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	if (!m_Hittable)
		return false;

	output_box = AABB(m_BoxMin, m_BoxMax);
	return true;
}
