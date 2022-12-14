#include "XyRect.h"

#include "Core/AABB.h"

XyRect::XyRect()
    : m_Pos(0.0f), m_K(0.0f), m_Material(nullptr)
{
}

XyRect::XyRect(const glm::mat2x2& pos, float k, const std::shared_ptr<Material>& material)
    : m_Pos(pos), m_K(k), m_Material(material)
{
}

bool XyRect::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
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
    output_box = AABB(Point3(m_Pos[0], m_K - 0.0001f), Point3(m_Pos[1], m_K + 0.0001f));
    return true;
}
