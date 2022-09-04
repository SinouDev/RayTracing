#include "Sphere.h"

#include "Core/Material/Lambertian.h"
#include "Core/AABB.h"

#include "glm/gtc/constants.hpp"

#include <cmath>

Sphere::Sphere(Point3& center, float r, std::shared_ptr<Material>& material)
    : m_Center(center), m_Radius(r), m_Material(material)
{
}

bool Sphere::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    Ray::Vec3 oc = ray.GetOrigin() - m_Center;
    float a = glm::dot(ray.GetDirection(), ray.GetDirection());
    float half_b = glm::dot(oc, ray.GetDirection());
    float c = glm::dot(oc, oc) - m_Radius * m_Radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f)
        return false;
    float discriminant_sqrt = glm::sqrt(discriminant);

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
    output_box = { m_Center - Vec3(m_Radius), m_Center + Vec3(m_Radius) };
    return true;
}

void Sphere::SetCenter(const Point3& center)
{
    m_Center = center;
}

void Sphere::GetSphereCoord(const Point3& p, Coord& coord)
{
    float theta = glm::acos(-p.y);
    float phi = std::atan2(-p.z, p.x) + glm::pi<float>();

    coord.s = phi / glm::two_pi<float>();
    coord.t = theta / glm::pi<float>();
}
