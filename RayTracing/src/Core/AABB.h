#pragma once

#define AABB_CLASS_DEFINED

#include "Utils/Math.h"

#include "glm/glm.hpp"

class Ray;

/// <summary>
/// 
/// </summary>
class AABB
{

public:

	/// <summary>
	/// 
	/// </summary>
	AABB();

	/// <summary>
	/// 
	/// </summary>
	/// <param name="a"></param>
	/// <param name="b"></param>
	AABB(const Utils::Math::Point3& a, const Utils::Math::Point3& b);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ray"></param>
	/// <param name="t_min"></param>
	/// <param name="t_max"></param>
	/// <returns></returns>
	bool Hit(const Ray& ray, float t_min, float t_max) const;


	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline Utils::Math::Point3 GetMin() const { return m_Minimum; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline Utils::Math::Point3 GetMax() const { return m_Maximum; }

private:

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Point3 m_Minimum;

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Point3 m_Maximum;

};