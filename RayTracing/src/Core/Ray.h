#pragma once

#include "Utils/Math.h"

class HittableObject;

/// <summary>
/// Ray class
/// </summary>
class Ray
{

public:

	/// <summary>
	/// 
	/// </summary>
	Ray() = default;

	/// <summary>
	/// 
	/// </summary>
	/// <param name="origin"></param>
	/// <param name="direction"></param>
	/// <param name="time"></param>
	/// <param name="backgroundColor"></param>
	/// <param name="backgroundColor1"></param>
	Ray(const Utils::Math::Point3& origin, const Utils::Math::Vec3& direction = Utils::Math::Vec3(0.0f), float time = 0.0f, const Utils::Math::Color3& backgroundColor = Utils::Math::Color3(0.5f, 0.7f, 1.0f), const Utils::Math::Color3& backgroundColor1 = Utils::Math::Color3(1.0f));

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Point3& GetOrigin() const { return m_Origin; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Vec3& GetDirection() const { return m_Direction; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Color3& GetRayBackgroundColor() { return m_RayBackgroundColor; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Color3& GetRayBackgroundColor1() { return m_RayBackgroundColor1; }
	inline const float GetTime() const { return m_Time; }
	//inline vec3& GetLightDir() { return m_LightDir; }

	/// <summary>
	/// 
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	Utils::Math::Point3 At(float t) const;

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	static inline bool& SimpleRayMode() { return s_SimpleRay; }

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ray"></param>
	/// <param name="backgroundColor"></param>
	/// <param name="list"></param>
	/// <param name="depth"></param>
	/// <returns></returns>
	static Utils::Math::Color4 RayColor(const Ray& ray, const Utils::Math::Color3& backgroundColor, const HittableObject& list, int32_t depth);

private:

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ray"></param>
	/// <returns></returns>
	friend Utils::Math::Color3 get_background(const Ray& ray);

private:

	/// <summary>
	/// 
	/// </summary>
	static inline bool s_SimpleRay = false;

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Point3 m_Origin;

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Vec3 m_Direction;

	/// <summary>
	/// 
	/// </summary>
	float m_Time;

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Color3 m_RayBackgroundColor;

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Color3 m_RayBackgroundColor1;
	//Utils::Math::Vec3 m_LightDir { 1.0f, 10.0f, 3.0f };

};