#include "YzRect.h"

#include "Core/AABB.h"

YzRect::YzRect()
    : m_Pos(0.0f), m_K(0.0f), m_Material(nullptr)
{
}

YzRect::YzRect(const glm::mat2x2& pos, float k, const std::shared_ptr<Material>& material)
    : m_Pos(pos), m_K(k), m_Material(material)
{
}

bool YzRect::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    float t = (m_K - ray.GetOrigin().x) / ray.GetDirection().x;

    if (t < min || t > max)
        return false;

    float y = ray.GetOrigin().y + t * ray.GetDirection().y;
    float z = ray.GetOrigin().z + t * ray.GetDirection().z;

    if (y < m_Pos[0].s || y > m_Pos[1].s || z < m_Pos[0].t || z > m_Pos[1].t)
        return false;

    hitRecord.coord.x = (y - m_Pos[0].s) / (m_Pos[1].s - m_Pos[0].s);
    hitRecord.coord.y = (z - m_Pos[0].t) / (m_Pos[1].t - m_Pos[0].t);

    hitRecord.t = t;
    Vec3 outward_normal(1.0f, 0.0f, 0.0f);
    hitRecord.set_face_normal(ray, outward_normal);
    hitRecord.material_ptr = m_Material;
    hitRecord.point = ray.At(t);

    return true;
}

bool YzRect::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    output_box = AABB(Point3(m_K - 0.0001f, m_Pos[0]), Point3(m_K + 0.0001f, m_Pos[1]));
    return true;
}
