#pragma once
#include "glm/glm.hpp"
#include <vector>

class Camera
{
public:

	Camera(float verticalFOV = 45.0f, float nearClip = 0.1f, float farClip = 100.0f, float aspectRatio = 0.0f, float aperture = 0.0f, float focusDistance = 0.0f, float lensRadius = 0.0f);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	void SetFOV(float verticalFOV);
	void SetNearClip(float nearClip);
	void SetFarClip(float farClip);

	void LookAt(glm::vec3& direction);
	void LookFrom(glm::vec3& position);

	const glm::mat4& GetProjection() const { return m_Projection; }
	const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	const glm::mat4& GetView() const { return m_View; }
	const glm::mat4& GetInverseView() const { return m_InverseView; }

	const glm::vec3& GetPosition() const { return m_Position; }
	const glm::vec3& GetDirection() const { return m_ForwardDirection; }

	const std::vector<glm::vec3>& GetRayDirections() const { return m_RayDirection; }

	bool* GetMultiThreadedRendering() { return &m_MultiThreadedRendering; }
	uint8_t* GetThreads() { return &m_Threads; }
	uint8_t inline GetWorkingThreadsCount() { return m_MultiThreadedRendering ? m_Threads : 1; }

	float GetRotationSpeed();

private:

	friend void async_ray_calc_func(Camera&, uint32_t, uint32_t, uint32_t);

private:

	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirection();

private:
	
	float m_VerticalFOV = 45.0f;
	float m_NearClip = 0.1f;
	float m_FarClip = 100.0f;

	float m_AspectRatio; // unused
	float m_Aperture;
	float m_FocusDistance;
	float m_LensRadius;

	glm::mat4 m_Projection{ 1.0f };
	glm::mat4 m_View{ 1.0f };
	glm::mat4 m_InverseProjection{ 1.0f };
	glm::mat4 m_InverseView{ 1.0f };
	
	float m_MouseSensetivity = 0.002f;
	float m_MoveSpeed = 5.0f;

	glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
	glm::vec3 m_ForwardDirection{ 0.0f, 0.0f, 0.0f };

	std::vector<glm::vec3> m_RayDirection;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	bool m_MultiThreadedRendering = false;
	uint8_t m_Threads = 11; // 11 threads seems to be the sweetspot on my own pc

};

