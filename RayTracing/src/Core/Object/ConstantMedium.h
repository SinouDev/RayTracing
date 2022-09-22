#pragma once

#include "HittableObject.h"

#include "Core/Material/Material.h"

#include <memory>

class Texture;

class ConstantMedium : public HittableObject
{
public:
	ConstantMedium(std::shared_ptr<HittableObject>& object, float d, std::shared_ptr<Texture>& texture);
	ConstantMedium(std::shared_ptr<HittableObject>& object, float d, Utils::Math::Color3 color);

	virtual bool Hit(const Ray& ray, float min, float max, HitRecord& hitRecord) const override;
	virtual bool BoundingBox(float _time0, float _time1, AABB& output_box) const override;

	virtual inline HittableObjectTypes GetType() const override { return HittableObjectTypes::CONTANT_MEDIUM; }

	inline std::shared_ptr<HittableObject>& GetBoundary() { return m_Boundary; }

	inline float& GetNegInverseDensity() { return m_NegInvDensity; }

	virtual inline Material* GetMaterial() override { return m_PhaseFunction->GetInstance(); }

	virtual inline std::shared_ptr<HittableObject> Clone() const override
	{
		return nullptr;
	}

private:

	std::shared_ptr<HittableObject> m_Boundary;
	std::shared_ptr<Material> m_PhaseFunction;
	float m_NegInvDensity;

};