#pragma once

#include "HittableObject.h"

#include "Core/AABB.h"

#include <memory>

class RotateZ : public HittableObject
{
public:

	RotateZ(std::shared_ptr<HittableObject>& object, float angle = 0.0f);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return HittableObjectTypes::ROTATE_Z; }

	inline std::shared_ptr<HittableObject>& GetObject() { return m_Object; }

	inline float& GetAngle() { return m_Angle; }

	void Rotate(float angle);
	void Rotate();

	virtual inline std::shared_ptr<HittableObject> Clone() const override
	{
		return nullptr;
	}

private:

	std::shared_ptr<HittableObject> m_Object;
	float m_Angle = 0.0f;
	float m_OldAngle = Utils::Math::infinity;
	float m_SinTheta = 0.0f;
	float m_CosTheta = 0.0f;
	bool m_HasBox = false;
	AABB m_Box;

};