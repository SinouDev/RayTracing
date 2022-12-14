#include "RotateZ.h"

#include "Utils/Utils.h"

RotateZ::RotateZ(std::shared_ptr<HittableObject>& object, float angle)
	: m_Object(object)
{
	float radians = glm::radians(angle);
	m_SinTheta = glm::sin(radians);
	m_CosTheta = glm::cos(radians);
	m_HasBox = object->BoundingBox(0.0f, 1.0f, m_Box);

	Point3 min(Utils::infinity, Utils::infinity, Utils::infinity);
	Point3 max(-Utils::infinity, -Utils::infinity, -Utils::infinity);

	for (int32_t i = 0; i < 2; i++)
	{
		for (int32_t j = 0; j < 2; j++)
		{
			for (int32_t k = 0; k < 2; k++)
			{
				float x = i * m_Box.GetMax().x + (1 - i) * m_Box.GetMin().x;
				float y = j * m_Box.GetMax().y + (1 - j) * m_Box.GetMin().y;
				float z = k * m_Box.GetMax().z + (1 - k) * m_Box.GetMin().z;

				float newx = m_CosTheta * x + m_SinTheta * y;
				float newy = -m_SinTheta * x + m_CosTheta * y;

				Vec3 tester(newx, newy, z);

				for (int32_t c = 0; c < 3; c++)
				{
					min[c] = glm::min(min[c], tester[c]);
					max[c] = glm::max(max[c], tester[c]);
				}

			}
		}
	}

	m_Box = AABB(min, max);

}

bool RotateZ::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	Vec3 origin = ray.GetOrigin();
	Vec3 direction = ray.GetDirection();

	origin.x = m_CosTheta * ray.GetOrigin().x - m_SinTheta * ray.GetOrigin().y;
	origin.y = m_SinTheta * ray.GetOrigin().x + m_CosTheta * ray.GetOrigin().y;

	direction.x = m_CosTheta * ray.GetDirection().x - m_SinTheta * ray.GetDirection().y;
	direction.y = m_SinTheta * ray.GetDirection().x + m_CosTheta * ray.GetDirection().y;

	Ray rotatedR(origin, direction, ray.GetTime());

	if (!m_Object->Hit(rotatedR, min, max, hitRecord))
		return false;

	Point3 point = hitRecord.point;
	Vec3 normal = hitRecord.normal;

	point.x = m_CosTheta * hitRecord.point.x + m_SinTheta * hitRecord.point.y;
	point.y = -m_SinTheta * hitRecord.point.x + m_CosTheta * hitRecord.point.y;

	normal.x = m_CosTheta * hitRecord.normal.x + m_SinTheta * hitRecord.normal.y;
	normal.y = -m_SinTheta * hitRecord.normal.x + m_CosTheta * hitRecord.normal.y;

	hitRecord.point = point;
	hitRecord.set_face_normal(rotatedR, normal);
}

bool RotateZ::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	output_box = m_Box;
	return m_HasBox;
}
