#include "MovingSphere.h"

#include "Core/Material/Material.h"
#include "Core/AABB.h"
#include "Utils/Utils.h"

using Utils::Math::Point3;
using Utils::Math::Vec3;

MovingSphere::MovingSphere()
    : m_Center0(0.0f), m_Center1(0.0f), m_Time0(0.0f), m_Time1(0.0f), m_Radius(0.0f), m_Material(nullptr)
{
    m_Name = "MovingSphere";
}

MovingSphere::MovingSphere(Point3& cen0, Point3& cen1, float _time0, float _time1, float radius, std::shared_ptr<Material>& material)
    : m_Center0(cen0), m_Center1(cen1), m_Time0(_time0), m_Time1(_time1), m_Radius(radius), m_Material(material)
{
    m_Name = "MovingSphere";
}

bool MovingSphere::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    if (!m_Hittable)
        return false;

    Vec3 oc = ray.GetOrigin() - GetCenter(ray.GetTime());
    float a = Utils::Math::Dot(ray.GetDirection(), ray.GetDirection());
    float half_b = Utils::Math::Dot(oc, ray.GetDirection());
    float c = Utils::Math::Dot(oc, oc) - m_Radius * m_Radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f)
        return false;
    float discriminant_sqrt = Utils::Math::Q_Rsqrt(discriminant);

    float root = (-half_b - discriminant_sqrt) / a;

    if (root < min || root > max)
    {
        root = (-half_b + discriminant_sqrt) / a;
        if (root < min || root > max)
            return false;
    }

    hitRecord.t = root;
    hitRecord.point = ray.At(hitRecord.t);
    Vec3 outward_normal = (hitRecord.point - GetCenter(ray.GetTime())) / m_Radius;
    hitRecord.set_face_normal(ray, outward_normal);
    hitRecord.material_ptr = m_Material;

	return true;
}

bool MovingSphere::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    if (!m_Hittable)
        return false;

    AABB box0{ GetCenter(_time0) - Vec3(m_Radius), GetCenter(_time0) + Vec3(m_Radius) };
    AABB box1{ GetCenter(_time1) - Vec3(m_Radius), GetCenter(_time1) + Vec3(m_Radius) };

    output_box = SurroundingBox(box0, box1);

    return true;
}

Point3 MovingSphere::GetCenter(float time) const
{
	return m_Center0 + ((time - m_Time0) / (m_Time1 - m_Time0)) * (m_Center1 - m_Center0);
}
