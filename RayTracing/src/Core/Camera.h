#pragma once

#include "glm/glm.hpp"
#include <vector>

#include "Utils/Math.h"

class Ray;

/// <summary>
/// 
/// </summary>
class Camera
{
public:
	
	/// <summary>
	/// 
	/// </summary>
	/// <param name="verticalFOV"></param>
	/// <param name="nearClip"></param>
	/// <param name="farClip"></param>
	/// <param name="aspectRatio"></param>
	/// <param name="aperture"></param>
	/// <param name="focusDistance"></param>
	/// <param name="_time0"></param>
	/// <param name="_time1"></param>
	Camera(float verticalFOV = 45.0f, float nearClip = 0.1f, float farClip = 100.0f, float aspectRatio = 16.0f / 9.0f, float aperture = 1.0f, float focusDistance = 1.0f, float _time0 = 0.0f, float _time1 = 0.0f);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ts"></param>
	virtual void OnUpdate(float ts);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	virtual void OnResize(uint32_t width, uint32_t height);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="nearClip"></param>
	virtual void SetNearClip(float nearClip);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="farClip"></param>
	virtual void SetFarClip(float farClip);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="direction"></param>
	virtual void LookAt(const Utils::Math::Vec3& direction);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="position"></param>
	virtual void LookFrom(const Utils::Math::Vec3& position);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="a"></param>
	virtual void SetAspectRatio(float a);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="a"></param>
	virtual void SetAperture(float a);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="d"></param>
	virtual void SetFocusDistance(float d);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="l"></param>
	virtual void SetLensRadius(float l);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="fov"></param>
	virtual void SetFOV(float fov);

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	virtual float GetRotationSpeed();

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Mat4& GetProjection() const { return m_Projection; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Mat4& GetInverseProjection() const { return m_InverseProjection; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Mat4& GetView() const { return m_View; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Mat4& GetInverseView() const { return m_InverseView; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Vec3& GetPosition() const { return m_Position; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const Utils::Math::Vec3& GetDirection() const { return m_ForwardDirection; }

	//inline const Directions& GetRayDirections() const { return m_RayDirection; }

	/// <summary>
	/// 
	/// </summary>
	/// <param name="uv"></param>
	/// <returns></returns>
	virtual Ray GetRay(const Utils::Math::Coord& uv) const;

	//bool* GetMultiThreadedRendering() { return &m_MultiThreadedRendering; }
	//uint8_t* GetThreads() { return &m_Threads; }
	//uint8_t inline GetWorkingThreadsCount() { return m_MultiThreadedRendering ? m_Threads : 1; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline float GetAspectRatio() { return m_AspectRatio; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline float GetAperture() { return m_Aperture; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline float GetFocusDistance() { return m_FocusDistance; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline float GetLensRadius() { return m_LensRadius; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline float& GetMoveSpeed() { return m_MoveSpeed; }

private:

	//friend void async_ray_calc_func(const Camera&, uint32_t, uint32_t, uint32_t);

protected:

	/// <summary>
	/// 
	/// </summary>
	virtual void RecalculateProjection();

	/// <summary>
	/// 
	/// </summary>
	virtual void RecalculateView();
	//void RecalculateRayDirection();

protected:

	//glm::vec3 m_LowerLeftCorner;
	//glm::vec3 m_Horizontal;
	//glm::vec3 m_Vertical;

	//glm::vec3 w;
	
	//glm::vec3 v;

	/// <summary>
	/// 
	/// </summary>
	float m_VerticalFOV;

	/// <summary>
	/// 
	/// </summary>
	float m_NearClip;

	/// <summary>
	/// 
	/// </summary>
	float m_FarClip;

	/// <summary>
	/// 
	/// </summary>
	float m_AspectRatio;

	/// <summary>
	/// 
	/// </summary>
	float m_Aperture;

	/// <summary>
	/// 
	/// </summary>
	float m_FocusDistance;

	/// <summary>
	/// 
	/// </summary>
	float m_Time0;

	/// <summary>
	/// 
	/// </summary>
	float m_Time1;


	/// <summary>
	/// 
	/// </summary>
	float m_LensRadius = 0.0f;


	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Mat2x3 m_ViewCoordMat{ 0.0f };

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Mat4 m_Projection{ 1.0f };

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Mat4 m_View{ 1.0f };

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Mat4 m_InverseProjection{ 1.0f };

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Mat4 m_InverseView{ 1.0f };


	/// <summary>
	/// 
	/// </summary>
	float m_MouseSensetivity = 0.002f;

	/// <summary>
	/// 
	/// </summary>
	float m_MoveSpeed = 5.0f;


	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Vec3 m_Position{ 0.0f, 0.0f, 0.0f };

	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Vec3 m_ForwardDirection{ 0.0f, 0.0f, 0.0f };

	//Directions m_RayDirection;


	/// <summary>
	/// 
	/// </summary>
	Utils::Math::Vec2 m_LastMousePosition{ 0.0f, 0.0f };


	/// <summary>
	/// 
	/// </summary>
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	//bool m_MultiThreadedRendering = false;
	//uint8_t m_Threads = 11; // 11 threads seems to be the sweetspot on my own pc

};

