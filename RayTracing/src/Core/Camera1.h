#pragma once

#pragma once
#include "glm/glm.hpp"
#include <vector>

class Ray;

class Camera
{

	/*
private:

	using point3 = glm::vec3;
	using vec3   = glm::vec3;

public:

	Camera() = default;

	Camera(const point3& lookFrom, const point3& lookAt, float fov, float aspectRatio);

	void OnUpdate(float ts);

	Ray GetRay(glm::vec2 coord);

	inline float GetAspectRatio() { return aspectRatio; }
	inline point3& GetPosition() { return origin; }
	inline point3& GetDirection() { return direction; }

private:

	point3 direction;
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	float aspectRatio;
	float vFov;

	*/
public:

	using DirectionSamples = std::vector<glm::vec3>;
	using Directions       = std::vector<DirectionSamples>;

public:
	
	Camera(float verticalFOV = 45.0f, float nearClip = 0.1f, float farClip = 100.0f, float aspectRatio = 16.0f / 9.0f, float aperture = 1.0f, float focusDistance = 1.0f);

	void OnUpdate(float ts);
	void OnResize(uint32_t width, uint32_t height);

	void SetNearClip(float nearClip);
	void SetFarClip(float farClip);

	void LookAt(glm::vec3& direction);
	void LookFrom(glm::vec3& position);

	//const glm::mat4& GetProjection() const { return m_Projection; }
	//const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
	//const glm::mat4& GetView() const { return m_View; }
	//const glm::mat4& GetInverseView() const { return m_InverseView; }

	inline const glm::vec3& GetPosition() const { return m_Position; }
	inline const glm::vec3& GetDirection() const { return m_ForwardDirection; }

	const Directions& GetRayDirections() const { return m_RayDirection; }

	Ray GetRay(const glm::vec2& uv) const;

	bool* GetMultiThreadedRendering() { return &m_MultiThreadedRendering; }
	uint8_t* GetThreads() { return &m_Threads; }
	uint8_t inline GetWorkingThreadsCount() { return m_MultiThreadedRendering ? m_Threads : 1; }

	inline float GetAspectRatio() { return m_AspectRatio; }
	inline float GetAperture() { return m_Aperture; }
	inline float GetFocusDistance() { return m_FocusDistance; }
	inline float GetLensRadius() { return m_LensRadius; }

	void SetAspectRatio(float a)
	{
		if (m_AspectRatio == a)
			return;
		m_AspectRatio = a;
		RecalculateView();
	}

	void SetAperture(float a) 
	{
		if (m_Aperture == a)
			return;
		m_Aperture = a;
		RecalculateView();
	}

	void SetFocusDistance(float d)
	{
		if (m_FocusDistance == d)
			return;
		m_FocusDistance = d;
		RecalculateView();
	}

	void SetLensRadius(float l) 
	{
		if (m_LensRadius == l)
			return;
		m_LensRadius = l;
		RecalculateView();
	}

	void SetFOV(float fov)
	{
		if (m_VerticalFOV == fov)
			return;
		m_VerticalFOV = fov;
		RecalculateView();
	}

	float GetRotationSpeed();

private:

	friend void async_ray_calc_func(Camera&, uint32_t, uint32_t, uint32_t);

private:

	void RecalculateProjection();
	void RecalculateView();
	void RecalculateRayDirection();

private:

	glm::vec3 m_LowerLeftCorner;
	glm::vec3 m_Horizontal;
	glm::vec3 m_Vertical;

	glm::vec3 w;
	glm::vec3 u;
	glm::vec3 v;

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

	Directions m_RayDirection;

	glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;

	bool m_MultiThreadedRendering = false;
	uint8_t m_Threads = 11; // 11 threads seems to be the sweetspot on my own pc

};

