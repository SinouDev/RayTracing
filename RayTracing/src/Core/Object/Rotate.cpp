#include "Rotate.h"

using Utils::Math::Min;
using Utils::Math::Max;

Rotate::Rotate(std::shared_ptr<HittableObject>& object, float angleX, float angleY, float angleZ)
	: Rotate(object, {angleX, angleY, angleZ})
{
}

Rotate::Rotate(std::shared_ptr<HittableObject>& object, const Utils::Math::Point3& angle)
	: m_Object(object), m_Angle(angle)
{
	m_Name = "Rotate";
	std::shared_ptr<HittableObject> rotateZ = std::make_shared<RotateZ>(object, angle.z);
	std::shared_ptr<HittableObject> rotateY = std::make_shared<RotateY>(rotateZ, angle.y);
	m_RotateX = std::make_shared<RotateX>(rotateY, angle.x);
	RotateAxis();
}

bool Rotate::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	return m_RotateX->GetObject()->GetInstance<RotateY>()->GetObject()->GetInstance<RotateZ>()->Hit(ray, min, max, hitRecord) ||
		m_RotateX->GetObject()->GetInstance<RotateY>()->Hit(ray, min, max, hitRecord) ||
		m_RotateX->Hit(ray, min, max, hitRecord);
	//return m_RotateX->Hit(ray, min, max, hitRecord)
	//	|| m_RotateY->Hit(ray, min, max, hitRecord) || m_RotateZ->Hit(ray, min, max, hitRecord);
}

bool Rotate::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	AABB boxX, boxY, boxZ;
	bool bound =
		//m_RotateX->BoundingBox(_time0, _time1, boxX) ||
		//m_RotateY->BoundingBox(_time0, _time1, boxX) ||
		//m_RotateZ->BoundingBox(_time0, _time1, boxX);
	m_RotateX->GetObject()->GetInstance<RotateY>()->GetObject()->GetInstance<RotateZ>()->BoundingBox(_time0, _time1, boxX) ||
	m_RotateX->GetObject()->GetInstance<RotateY>()->BoundingBox(_time0, _time1, boxX) ||
	m_RotateX->BoundingBox(_time0, _time1, boxX);


	output_box = boxX;// AABB(Min(boxX.GetMin(), Min(boxY.GetMin(), boxZ.GetMin())), Max(boxX.GetMax(), Max(boxY.GetMax(), boxZ.GetMax())));

	return bound;
}

void Rotate::RotateAxis()
{
	using Utils::Math::Degrees;
	//m_RotateZ->Rotate();
	//m_RotateY->Rotate();
	m_RotateX->GetObject()->GetInstance<RotateY>()->GetObject()->GetInstance<RotateZ>()->Rotate(Degrees(m_Angle.x));
	m_RotateX->GetObject()->GetInstance<RotateY>()->Rotate(Degrees(m_Angle.y));
	m_RotateX->Rotate(Degrees(m_Angle.z));
}

void Rotate::RotateAxis(float angleX, float angleY, float angleZ)
{
	m_Angle.x = angleX;
	m_Angle.y = angleY;
	m_Angle.z = angleZ;
	RotateAxis();
}

void Rotate::RotateAxis(const Utils::Math::Point3& angle)
{
	m_Angle = angle;
	RotateAxis();
}
