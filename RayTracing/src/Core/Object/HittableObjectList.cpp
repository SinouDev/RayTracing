#include "HittableObjectList.h"

#include "Core/AABB.h"

HittableObjectList::HittableObjectList(std::shared_ptr<HittableObject>& object)
{
    Add(object);
}

void HittableObjectList::Clear()
{
    m_Objects.clear();
}

void HittableObjectList::Add(const std::shared_ptr<HittableObject>& object)
{
    m_Objects.emplace_back(object);
}

bool HittableObjectList::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    HitRecord temp_hit_record = hitRecord;
    bool hit_anything = false;
    float closest_so_far = max;

    for (const auto& object : m_Objects)
    {
        if (object->Hit(ray, min, closest_so_far, temp_hit_record))
        {
            
            hit_anything = true;
            closest_so_far = temp_hit_record.t;
            hitRecord = temp_hit_record;
        }
    }

    return hit_anything;
}

bool HittableObjectList::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    if (m_Objects.empty())
        return false;

    AABB temp_box;
    bool firt_box = true;

    return true;
}
