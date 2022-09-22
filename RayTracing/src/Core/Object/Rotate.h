#pragma once

#include "RotateX.h"
#include "RotateY.h"
#include "RotateZ.h"

class Rotate : public HittableObject
{
public:

	Rotate(std::shared_ptr<HittableObject>& object, float angleX = 0.0f, float angleY = 0.0f, float angleZ = 0.0f);
	Rotate(std::shared_ptr<HittableObject>& object, const Utils::Math::Point3& angle = Utils::Math::Point3(0.0f));

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return HittableObjectTypes::ROTATE; }

	inline std::shared_ptr<HittableObject>& GetObject() { return m_Object; }

	void RotateAxis();
	void RotateAxis(float angleX, float angleY, float angleZ);
	void RotateAxis(const Utils::Math::Point3& angle);

	inline Utils::Math::Point3& GetAngle() { return m_Angle; }

	inline RotateX* GetRotateX() { return m_RotateX->GetInstance<RotateX>(); }
	inline RotateY* GetRotateY() { return m_RotateX->GetObject()->GetInstance<RotateY>(); }
	inline RotateZ* GetRotateZ() { return GetRotateY()->GetObject()->GetInstance<RotateZ>(); }
	//inline std::shared_ptr<RotateZ>& GetRotateZ() { return m_RotateZ; }

	virtual inline std::shared_ptr<HittableObject> Clone() const override
	{
		return nullptr;
	}

private:

	std::shared_ptr<HittableObject> m_Object;
	Utils::Math::Point3 m_Angle;
	std::shared_ptr<RotateX> m_RotateX;
	//std::shared_ptr<RotateY> m_RotateY;
	//std::shared_ptr<RotateZ> m_RotateZ;

};