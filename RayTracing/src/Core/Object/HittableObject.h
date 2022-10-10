#pragma once

#include "HittableObjectBase.h"

#include "Core/BaseObject.h"
#include "Core/Material/Material.h"
#include "Core/AABB.h"

#include "Utils/Math.h"
#include "Utils/Reference.h"

#include <string>

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

	virtual inline bool IsFeatureObject() { return m_FeatureObject; }

	virtual inline bool& GetHittable() { return m_Hittable; }

	virtual inline void SetName(const char* name) override
	{
		m_Name = name;
	}

	virtual inline void SetName(const std::string& name)
	{
		m_Name = name;
	}

	virtual inline Utils::Math::Vec3& GetObjectTranslate() { return m_ObjectTranslate; }
	virtual inline Utils::Math::Vec3& GetObjectRotate() { return m_ObjectRotate; }
	virtual inline Utils::Math::Vec3& GetObjectScale() { return m_ObjectScale; }

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

	static inline AABB SurroundingBox(const AABB& box0, const AABB& box1)
	{
		// changed to small_b to not conflict with "#define small char" defined in <rpcndr.h>
		Utils::Math::Vec3 small_b(Utils::Math::Min(box0.GetMin(), box1.GetMin()));
		Utils::Math::Vec3 big_b(Utils::Math::Max(box0.GetMax(), box1.GetMax()));

		return AABB(small_b, big_b);
	}

protected:

	Utils::Math::Vec3 m_ObjectTranslate{ 0.0f };
	Utils::Math::Vec3 m_ObjectRotate{ 0.0f };
	Utils::Math::Vec3 m_ObjectScale{ 0.0f };

	std::string m_Name = "";
	bool m_Hittable = true;
	bool m_FeatureObject = false;
};