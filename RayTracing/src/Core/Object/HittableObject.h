#pragma once

#include "HittableObjectBase.h"

#include <string>

// Any new added hittable object subclass must be defined here or it will not be recognized in the tree.
enum HittableObjectTypes : uint32_t {

	UNKNOWN        = 0x00,

	OBJECT_LIST    = 0x02,

	SPHERE         = 0x04,
	MOVING_SPHERE,

	BOX            = 0x07,
	XY_RECT,
	XZ_RECT,
	YZ_RECT,

	TRANSLATE      = 0x0C,
	ROTATE_X,
	ROTATE_Y,
	ROTATE_Z,

	CONTANT_MEDIUM = 0x11,

	BVH_NODE       = 0x13
	
};


class HittableObject : public HittableObjectBase<HittableObjectTypes>
{
public:

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override = 0;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override = 0;

	template<typename T>
	inline T* GetInstance() 
	{
		static_assert(std::is_base_of<HittableObject, T>(), "This is not a subclass of HittableObject");
		return dynamic_cast<T*>(this);
	}

	inline HittableObject* GetInstance() { return this; }

	virtual inline HittableObjectTypes GetType() const override { return UNKNOWN; }

	virtual inline const char* GetName() const override { return m_Name.c_str(); }

	virtual inline void SetName(const char* name) override
	{
		m_Name = name;
	}

	virtual inline void SetName(const std::string& name)
	{
		m_Name = name;
	}

protected:

	std::string m_Name = "";
};