#include "XyRect.h"

#include "Core/AABB.h"

using Utils::Math::Point3;
using Utils::Math::Vec3;
using Utils::Math::Mat2x2;

XyRect::XyRect()
    : XyRect(Mat2x2(0.0f), 0.0f, nullptr)
{

}

XyRect::XyRect(const Mat2x2& pos, float k, const std::shared_ptr<Material>& material)
    : m_Pos(pos), m_K(k), m_Material(material)
{
    m_Name = "XyRect";
}

bool XyRect::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    if (!m_Hittable)
        return false;

    float t = (m_K - ray.GetOrigin().z) / ray.GetDirection().z;

    if (t < min || t > max)
        return false;

    float x = ray.GetOrigin().x + t * ray.GetDirection().x;
    float y = ray.GetOrigin().y + t * ray.GetDirection().y;

    if (x < m_Pos[0].x || x > m_Pos[1].x || y < m_Pos[0].y || y > m_Pos[1].y)
        return false;

    hitRecord.coord.x = (x - m_Pos[0].x) / (m_Pos[1].x - m_Pos[0].x);
    hitRecord.coord.y = (y - m_Pos[0].y) / (m_Pos[1].y - m_Pos[0].y);

    hitRecord.t = t;
    Vec3 outward_normal(0.0f, 0.0f, 1.0f);
    hitRecord.set_face_normal(ray, outward_normal);
    hitRecord.material_ptr = m_Material;
    hitRecord.point = ray.At(t);

    return true;
}

bool XyRect::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    if (!m_Hittable)
        return false;

    output_box = AABB(Point3(m_Pos[0], m_K - 0.0001f), Point3(m_Pos[1], m_K + 0.0001f));
    return true;
}
