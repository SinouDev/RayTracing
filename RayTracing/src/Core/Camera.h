#pragma once

#pragma once
#include "glm/glm.hpp"
#include <vector>

class Ray;

class Camera
{
public:
	
	Camera(float verticalFOV = 45.0f, float nearClip = 0.1f, float farClip = 100.0f, float aspectRatio = 16.0f / 9.0f, float aperture = 1.0f, float focusDistance = 1.0f, float _time0 = 0.0f, float _time1 = 0.0f);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);
	void SetNearClip(float nearClip);
	void SetFarClip(float farClip);
	void LookAt(const glm::vec3& direction);
	void LookFrom(const glm::vec3& position);
	void SetAspectRatio(float a);
	void SetAperture(float a);
	void SetFocusDistance(float d);
	void SetLensRadius(float l);
	void SetFOV(float fov);
	float GetRotationSpeed();

	inline const glm::mat4& GetProjection() const { return m_Projection; }
	inline const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	inline const glm::mat4& GetView() const { return m_View; }
	inline const glm::mat4& GetInverseView() const { return m_InverseView; }
	inline const glm::vec3& GetPosition() const { return m_Position; }
	inline const glm::vec3& GetDirection() const { return m_ForwardDirection; }

	//inline const Directions& GetRayDirections() const { return m_RayDirection; }

	Ray GetRay(const glm::vec2& uv) const;

	//bool* GetMultiThreadedRendering() { return &m_MultiThreadedRendering; }
	//uint8_t* GetThreads() { return &m_Threads; }
	//uint8_t inline GetWorkingThreadsCount() { return m_MultiThreadedRendering ? m_Threads : 1; }

	inline float GetAspectRatio() { return m_AspectRatio; }
	inline float GetAperture() { return m_Aperture; }
	inline float GetFocusDistance() { return m_FocusDistance; }
	inline float GetLensRadius() { return m_LensRadius; }
	inline float& GetMoveSpeed() { return m_MoveSpeed; }

private:

	//friend void async_ray_calc_func(const Camera&, uint32_t, uint32_t, uint32_t);

private:

	void RecalculateProjection();
	void RecalculateView();
	//void RecalculateRayDirection();

private:

	//glm::vec3 m_LowerLeftCorner;
	//glm::vec3 m_Horizontal;
	//glm::vec3 m_Vertical;

	//glm::vec3 w;
	
	//glm::vec3 v;

	float m_VerticalFOV;
	float m_NearClip;
	float m_FarClip;
	float m_AspectRatio;
	float m_Aperture;
	float m_FocusDistance;
	float m_Time0;
	float m_Time1;

	float m_LensRadius = 0.0f;

	glm::mat2x3 m_ViewCoordMat{ 0.0f };

	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };

	float m_MouseSensetivity = 0.002f;
	float m_MoveSpeed = 5.0f;

	glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
	glm::vec3 m_ForwardDirection{ 0.0f, 0.0f, 0.0f };

	//Directions m_RayDirection;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	//bool m_MultiThreadedRendering = false;
	//uint8_t m_Threads = 11; // 11 threads seems to be the sweetspot on my own pc

};

