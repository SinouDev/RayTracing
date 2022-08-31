#include "HittableObjectList.h"

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

bool HittableObjectList::Hit(Ray& ray, double min, double max, HitRecord& hitRecord) const
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
