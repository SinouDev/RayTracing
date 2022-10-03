#include "Sphere.h"

#include "Core/Material/Lambertian.h"
#include "Core/AABB.h"

using Utils::Math::Point3;
using Utils::Math::Vec3;
using Utils::Math::Coord;

Sphere::Sphere(Point3& center, float r, std::shared_ptr<Material>& material)
    : m_Center(center), m_Radius(r), m_Material(material)
{
    m_Name = "Sphere";
}

bool Sphere::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    if (!m_Hittable)
        return false;

    Vec3 oc = ray.GetOrigin() - m_Center;
    float a = Utils::Math::Dot(ray.GetDirection(), ray.GetDirection());
    float half_b = Utils::Math::Dot(oc, ray.GetDirection());
    float c = Utils::Math::Dot(oc, oc) - m_Radius * m_Radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f)
        return false;
    float discriminant_sqrt = Utils::Math::Q_Sqrt(discriminant);

    float root = (-half_b - discriminant_sqrt) / a;

    if (root < min || root > max)
    {
        root = (-half_b + discriminant_sqrt) / a;
        if (root < min || root > max)
            return false;
    }

    hitRecord.t = root;
    hitRecord.point = ray.At(hitRecord.t);
    Vec3 outward_normal = (hitRecord.point - m_Center) / m_Radius;
    hitRecord.set_face_normal(ray, outward_normal);
    GetSphereCoord(outward_normal, hitRecord.coord);
    hitRecord.material_ptr = m_Material;

    return true;
}

bool Sphere::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    if (!m_Hittable)
        return false;

    output_box = { m_Center - Vec3(m_Radius), m_Center + Vec3(m_Radius) };
    return true;
}

void Sphere::SetCenter(const Point3& center)
{
    m_Center = center;
}

void Sphere::GetSphereCoord(const Point3& p, Coord& coord)
{
    float theta = Utils::Math::Acos(-p.y);
    float phi = Utils::Math::Atan2(-p.z, p.x) + Utils::Math::Pi<float>();

    coord.s = phi / Utils::Math::TwoPi<float>();
    coord.t = theta / Utils::Math::Pi<float>();
}
