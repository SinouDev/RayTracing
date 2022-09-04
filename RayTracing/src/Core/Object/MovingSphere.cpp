#include "MovingSphere.h"

#include "Core/Material/Material.h"
#include "Core/AABB.h"
#include "Core/Utils.h"

MovingSphere::MovingSphere()
{
}

MovingSphere::MovingSphere(point3& cen0, point3& cen1, float _time0, float _time1, float radius, std::shared_ptr<Material>& material)
    : m_Center0(cen0), m_Center1(cen1), m_Time0(_time0), m_Time1(_time1), m_Radius(radius), m_Material(material)
{
}

bool MovingSphere::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    Ray::vec3 oc = ray.GetOrigin() - GetCenter(ray.GetTime());
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
    vec3 outward_normal = (hitRecord.point - GetCenter(ray.GetTime())) / m_Radius;
    hitRecord.set_face_normal(ray, outward_normal);
    hitRecord.material_ptr = m_Material;

	return true;
}

bool MovingSphere::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    AABB box0{ GetCenter(_time0) - vec3(m_Radius), GetCenter(_time0) + vec3(m_Radius) };
    AABB box1{ GetCenter(_time1) - vec3(m_Radius), GetCenter(_time1) + vec3(m_Radius) };

    output_box = Utils::SurroundingBox(box0, box1);

    return true;
}

HittableObject::point3 MovingSphere::GetCenter(float time) const
{
	return m_Center0 + ((time - m_Time0) / (m_Time1 - m_Time0)) * (m_Center1 - m_Center0);
}
