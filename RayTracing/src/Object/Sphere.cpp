#include "Sphere.h"

#include "../Material/Lambertian.h"

Sphere::Sphere(point3& center, float r, std::shared_ptr<Material>& material)
    : m_Center(center), m_Radius(r), m_Material(material)
{
}

bool Sphere::Hit(Ray& ray, double min, double max, HitRecord& hitRecord) const
{
    Ray::vec3 oc = ray.GetOrigin() - m_Center;
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
    vec3 outward_normal = (hitRecord.point - m_Center) / m_Radius;
    hitRecord.set_face_normal(ray, outward_normal);
    hitRecord.material_ptr = m_Material;

    return true;
}

void Sphere::SetCenter(const point3& center)
{
    m_Center = center;
}
