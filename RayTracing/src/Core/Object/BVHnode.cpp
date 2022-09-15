#include "BVHnode.h"

#include "HittableObjectList.h"

#include "Utils/Utils.h"
#include "Utils/Random.h"

#include <algorithm>
#include <iostream>

bool box_compare_x(const BVHnode::HittablePtr& a, const BVHnode::HittablePtr& b)
{
    return BVHnode::BoxCompare(a, b, 0);
}
bool box_compare_y(const BVHnode::HittablePtr& a, const BVHnode::HittablePtr& b)
{
    return BVHnode::BoxCompare(a, b, 1);
}
bool box_compare_z(const BVHnode::HittablePtr& a, const BVHnode::HittablePtr& b)
{
    return BVHnode::BoxCompare(a, b, 2);
}

BVHnode::BVHnode()
    : m_Left(nullptr), m_Right(nullptr)
{
}

BVHnode::BVHnode(const HittableObjectList& list, float _time0, float _time1)
    : BVHnode(list.m_Objects, 0, list.m_Objects.size(), _time0, _time1)
{
}

BVHnode::BVHnode(const HittableList& srcObjects, size_t start, size_t end, float _time0, float _time1)
{
    auto objects = srcObjects;
    int32_t axis = Utils::Random::RandomInt(0, 2);

    auto comparator = (axis == 0) ? box_compare_x : (axis == 1) ? box_compare_y : box_compare_z;

    size_t objectSpan = end - start;

    if (objectSpan == 1)
    {
        m_Left = m_Right = objects[start];
    }
    else if (objectSpan == 2)
    {
        if (comparator(objects[start], objects[start + 1]))
        {
            m_Left = objects[start];
            m_Right = objects[start + 1];
        }
        else
        {
            m_Left = objects[start + 1];
            m_Right = objects[start];
        }
    }
    else
    {
        std::sort(objects.begin() + start, objects.begin() + end, comparator);
        auto mid = start + objectSpan / 2;

        m_Left = std::make_shared<BVHnode>(objects, start, mid, _time0, _time1);
        m_Right = std::make_shared<BVHnode>(objects, mid, end, _time0, _time1);

    }

    AABB boxLeft, boxRight;
    if (!m_Left->BoundingBox(_time0, _time1, boxLeft) || !m_Right->BoundingBox(_time0, _time1, boxRight))
        std::cerr << "No bounding box in BVHnode constructor\n";

    m_Box = Utils::SurroundingBox(boxLeft, boxRight);

}

bool BVHnode::Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const
{
    if(!m_Box.Hit(ray, min, max))
        return false;

    bool hit_left = m_Left->Hit(ray, min, max, hitRecord);
    bool hit_right = m_Right->Hit(ray, min, hit_left ? hitRecord.t : max, hitRecord);

    return hit_left || hit_right;
}

bool BVHnode::BoundingBox(float _time0, float _time1, AABB& output_box) const
{
    output_box = m_Box;
    return true;
}

bool BVHnode::BoxCompare(const HittablePtr& a, const HittablePtr& b, int32_t axis)
{
    AABB boxA;
    AABB boxB;

    if (!a->BoundingBox(0, 0, boxA) || !b->BoundingBox(0, 0, boxB))
        std::cerr << "No bounding box in BVHnode constructor\n";

    return boxA.GetMin()[axis] < boxB.GetMin()[axis];
}