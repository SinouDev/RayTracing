#pragma once

#include "HittableObjectBase.h"

#include "Core/BaseObject.h"
#include "Core/Material/Material.h"

// Any new added hittable object subclass must be defined here or it will not be recognized in the tree.
enum HittableObjectTypes : uint32_t {

	UNKNOWN_OBJECT = 0x00,

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
	ROTATE,

	CONTANT_MEDIUM = 0x12,

	BVH_NODE       = 0x14
	
};


class HittableObject : public HittableObjectBase<HittableObjectTypes>, public virtual BaseObject<HittableObject>
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

	virtual inline Material* GetMaterial() { return nullptr; }

	inline HittableObject* GetInstance() { return this; }

	virtual inline HittableObjectTypes GetType() const override { return HittableObjectTypes::UNKNOWN_OBJECT; }

	virtual inline const char* GetName() const override { return m_Name.c_str(); }

	virtual inline void SetHittable(bool hittable = true) { m_Hittable = hittable; }

	virtual inline bool IsHittable() { return m_Hittable; }

	virtual inline bool& GetHittable() { return m_Hittable; }

	virtual inline void SetName(const char* name) override
	{
		m_Name = name;
	}

	virtual inline void SetName(const std::string& name)
	{
		m_Name = name;
	}

	static inline const char* GetTypeName(HittableObjectTypes type)
	{
		switch (type)
		{
			case OBJECT_LIST:
				return "List";
			case SPHERE:
				return "Sphere";
			case MOVING_SPHERE:
				return "Moving Sphere";
			case BOX:
				return "Box";
			case XY_RECT:
				return "XY Rect";
			case XZ_RECT:
				return "XZ Rect";
			case YZ_RECT:
				return "YZ Rect";
			case TRANSLATE:
				return "Translate";
			case ROTATE_X:
				return "Rotate X";
			case ROTATE_Y:
				return "Rotate Y";
			case ROTATE_Z:
				return "Rotate Z";
			case ROTATE:
				return "Rotate";
			case CONTANT_MEDIUM:
				return "Constant Medium";
			case BVH_NODE:
				return "BVH Node";
			case UNKNOWN_OBJECT:
			default:
				return "Unknown";
		}
	}

protected:

	std::string m_Name = "";
	bool m_Hittable = true;
};