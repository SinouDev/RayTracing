#include "RotateX.h"

#include "Core/Utils.h"

RotateX::RotateX(std::shared_ptr<HittableObject>& object, float angle)
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

				float newy = m_CosTheta * y + m_SinTheta * z;
				float newz = -m_SinTheta * y + m_CosTheta * z;

				Vec3 tester(x, newy, newz);

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

bool RotateX::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
	Vec3 origin = ray.GetOrigin();
	Vec3 direction = ray.GetDirection();

	origin.y = m_CosTheta * ray.GetOrigin().y - m_SinTheta * ray.GetOrigin().z;
	origin.z = m_SinTheta * ray.GetOrigin().y + m_CosTheta * ray.GetOrigin().z;

	direction.y = m_CosTheta * ray.GetDirection().y - m_SinTheta * ray.GetDirection().z;
	direction.z = m_SinTheta * ray.GetDirection().y + m_CosTheta * ray.GetDirection().z;

	Ray rotatedR(origin, direction, ray.GetTime());

	if (!m_Object->Hit(rotatedR, min, max, hitRecord))
		return false;

	Point3 point = hitRecord.point;
	Vec3 normal = hitRecord.normal;

	point.y = m_CosTheta * hitRecord.point.y + m_SinTheta * hitRecord.point.z;
	point.z = -m_SinTheta * hitRecord.point.y + m_CosTheta * hitRecord.point.z;

	normal.y = m_CosTheta * hitRecord.normal.y + m_SinTheta * hitRecord.normal.z;
	normal.z = -m_SinTheta * hitRecord.normal.y + m_CosTheta * hitRecord.normal.z;

	hitRecord.point = point;
	hitRecord.set_face_normal(rotatedR, normal);

	return true;
}

bool RotateX::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
	output_box = m_Box;
	return m_HasBox;
}
