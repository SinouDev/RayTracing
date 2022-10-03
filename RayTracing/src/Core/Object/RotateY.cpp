#include "RotateY.h"

using Utils::Math::Point3;
using Utils::Math::Vec3;

RotateY::RotateY(std::shared_ptr<HittableObject>& object, float angle)
	: m_Object(object)
{
	m_Name = "RotateY";
	m_FeatureObject = true;
	Rotate(angle);
}

bool RotateY::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	if (!m_Hittable)
		return false;

	Vec3 origin = ray.GetOrigin();
	Vec3 direction = ray.GetDirection();

	origin.x = m_CosTheta * ray.GetOrigin().x - m_SinTheta * ray.GetOrigin().z;
	origin.z = m_SinTheta * ray.GetOrigin().x + m_CosTheta * ray.GetOrigin().z;

	direction.x = m_CosTheta * ray.GetDirection().x - m_SinTheta * ray.GetDirection().z;
	direction.z = m_SinTheta * ray.GetDirection().x + m_CosTheta * ray.GetDirection().z;

	Ray rotatedR(origin, direction, ray.GetTime());
	
	if(!m_Object->Hit(rotatedR, min, max, hitRecord))
		return false;

	Point3 point = hitRecord.point;
	Vec3 normal = hitRecord.normal;

	point.x = m_CosTheta * hitRecord.point.x + m_SinTheta * hitRecord.point.z;
	point.z = -m_SinTheta * hitRecord.point.x + m_CosTheta * hitRecord.point.z;

	normal.x = m_CosTheta * hitRecord.normal.x + m_SinTheta * hitRecord.normal.z;
	normal.z = -m_SinTheta * hitRecord.normal.x + m_CosTheta * hitRecord.normal.z;

	hitRecord.point = point;
	hitRecord.set_face_normal(rotatedR, normal);

	return true;
}

bool RotateY::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	if (!m_Hittable)
		return false;

	output_box = m_Box;
	return m_HasBox;
}

void RotateY::Rotate(float angle)
{
	m_Angle = Utils::Math::Radians(angle);
	Rotate();
}

void RotateY::Rotate()
{
	if (m_OldAngle == m_Angle)
		return;

	//float radians = Utils::Math::Radians(angle);
	m_SinTheta = Utils::Math::Sin(m_Angle);
	m_CosTheta = Utils::Math::Cos(m_Angle);
	m_HasBox = m_Object->BoundingBox(0.0f, 1.0f, m_Box);

	Point3 min(Utils::Math::infinity, Utils::Math::infinity, Utils::Math::infinity);
	Point3 max(-Utils::Math::infinity, -Utils::Math::infinity, -Utils::Math::infinity);

	for (int32_t i = 0; i < 2; i++)
	{
		for (int32_t j = 0; j < 2; j++)
		{
			for (int32_t k = 0; k < 2; k++)
			{
				float x = i * m_Box.GetMax().x + (1 - i) * m_Box.GetMin().x;
				float y = j * m_Box.GetMax().y + (1 - j) * m_Box.GetMin().y;
				float z = k * m_Box.GetMax().z + (1 - k) * m_Box.GetMin().z;

				float newx = m_CosTheta * x + m_SinTheta * z;
				float newz = -m_SinTheta * x + m_CosTheta * z;

				Vec3 tester(newx, y, newz);

				for (int32_t c = 0; c < 3; c++)
				{
					min[c] = Utils::Math::Min(min[c], tester[c]);
					max[c] = Utils::Math::Max(max[c], tester[c]);
				}

			}
		}
	}

	m_Box = AABB(min, max);

	m_OldAngle = m_Angle;
}
