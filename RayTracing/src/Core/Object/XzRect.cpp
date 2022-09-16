#include "XzRect.h"

#include "Core/AABB.h"

using Utils::Math::Point3;
using Utils::Math::Vec3;
using Utils::Math::Mat2x2;

XzRect::XzRect()
    : XzRect(Mat2x2(0.0f), 0.0f, nullptr)
{
}

XzRect::XzRect(const Mat2x2& pos, float k, const std::shared_ptr<Material>& material)
    : m_Pos(pos), m_K(k), m_Material(material)
{
    m_Name = "XzRect";
}

bool XzRect::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    float t = (m_K - ray.GetOrigin().y) / ray.GetDirection().y;

    if (t < min || t > max)
        return false;

    float x = ray.GetOrigin().x + t * ray.GetDirection().x;
    float z = ray.GetOrigin().z + t * ray.GetDirection().z;

    if (x < m_Pos[0].s || x > m_Pos[1].s || z < m_Pos[0].t || z > m_Pos[1].t)
        return false;

    hitRecord.coord.x = (x - m_Pos[0].s) / (m_Pos[1].s - m_Pos[0].s);
    hitRecord.coord.y = (z - m_Pos[0].t) / (m_Pos[1].t - m_Pos[0].t);

    hitRecord.t = t;
    Vec3 outward_normal(0.0f, 1.0f, 0.0f);
    hitRecord.set_face_normal(ray, outward_normal);
    hitRecord.material_ptr = m_Material;
    hitRecord.point = ray.At(t);

    return true;
}

bool XzRect::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    output_box = AABB(Point3(m_Pos[0].s, m_K - 0.0001f, m_Pos[0].t), Point3(m_Pos[1].s, m_K + 0.0001f, m_Pos[1].t));
    return true;
}
