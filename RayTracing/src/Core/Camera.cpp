#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "Walnut/Input/Input.h"
#include "Walnut/Input/KeyCodes.h"

#include "Ray.h"

#include <thread>

constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);

glm::vec3 right;
glm::vec3 p_up;
glm::vec3 v;

Camera::Camera(float verticalFOV, float nearClip, float farClip, float aspectRatio, float aperture, float focusDistance, float lensRadius)
	: m_VerticalFOV(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip), m_AspectRatio(aspectRatio), m_Aperture(aperture), m_FocusDistance(focusDistance), m_LensRadius(lensRadius)
{
	//m_ForwardDirection = glm::vec3(0.0f, 0.0f, -1.0f);
	//m_Position = glm::vec3(0.0f, 0.0f, 3.0f);

	right = glm::normalize(glm::cross(m_ForwardDirection - m_Position, upDirection));
	p_up = glm::normalize(glm::cross(m_ForwardDirection - m_Position, right));


}

void Camera::OnUpdate(float ts)
{
	glm::vec2 mousePosition = Walnut::Input::GetMousePosition();
	glm::vec2 delta = (mousePosition - m_LastMousePosition) * m_MouseSensetivity;
	m_LastMousePosition = mousePosition;

	if (!Walnut::Input::IsMouseButtonDown(Walnut::MouseButton::Right))
	{
		Walnut::Input::SetCursorMode(Walnut::CursorMode::Normal);
		return;
	}

	Walnut::Input::SetCursorMode(Walnut::CursorMode::Locked);

	bool mouseMoved = false;

	glm::vec3 rightDirection = glm::cross(m_ForwardDirection, upDirection);
	
	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::W))
	{
		m_Position += m_ForwardDirection * m_MoveSpeed * ts;
		mouseMoved = true;
	}
	else if (Walnut::Input::IsKeyDown(Walnut::KeyCode::S))
	{
		m_Position -= m_ForwardDirection * m_MoveSpeed * ts;
		mouseMoved = true;
	}

	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::A))
	{
		m_Position -= rightDirection * m_MoveSpeed * ts;
		mouseMoved = true;
	}
	else if (Walnut::Input::IsKeyDown(Walnut::KeyCode::D))
	{
		m_Position += rightDirection * m_MoveSpeed * ts;
		mouseMoved = true;
	}

	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::Q))
	{
		m_Position -= upDirection * m_MoveSpeed * ts;
		mouseMoved = true;
	}
	else if (Walnut::Input::IsKeyDown(Walnut::KeyCode::E))
	{
		m_Position += upDirection * m_MoveSpeed * ts;
		mouseMoved = true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection), glm::angleAxis(-yawDelta, upDirection)));
		m_ForwardDirection = glm::rotate(q, m_ForwardDirection);

		mouseMoved = true;

	}

	if (mouseMoved)
	{
		RecalculateView();
		//RecalculateRayDirection();
	}

}

void Camera::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_ViewportWidth && height == m_ViewportHeight)
	{
		return;
	}

	m_AspectRatio = width / height;

	m_ViewportWidth = width;
	m_ViewportHeight = height;

	RecalculateProjection();
	RecalculateView();
	//RecalculateRayDirection();

}

void Camera::SetFOV(float verticalFOV)
{
	if (verticalFOV == m_VerticalFOV)
		return;
	m_VerticalFOV = verticalFOV;

	RecalculateProjection();
	RecalculateView();
	//RecalculateRayDirection();
}

void Camera::SetNearClip(float nearClip)
{
	if (nearClip == m_NearClip)
		return;
	m_NearClip = nearClip;

	RecalculateProjection();
	RecalculateView();
	//RecalculateRayDirection();
}

void Camera::SetFarClip(float farClip)
{
	if (farClip == m_FarClip)
		return;
	m_FarClip = farClip;

	RecalculateProjection();
	RecalculateView();
	//RecalculateRayDirection();
}

void Camera::LookAt(glm::vec3& direction)
{
	m_ForwardDirection = direction;
	RecalculateView();
}

void Camera::LookFrom(glm::vec3& position)
{
	m_Position = position;
	RecalculateView();
}

Ray Camera::GetRay(glm::vec2 coord)
{
	glm::vec4 target = m_InverseProjection * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
	glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f));
	return Ray(m_Position, rayDirection);
}

float Camera::GetRotationSpeed()
{
	return 0.3f;
}

void Camera::RecalculateProjection()
{
	m_Projection = glm::perspectiveFov(glm::radians(m_VerticalFOV), (float)m_ViewportWidth, (float)m_ViewportHeight, m_NearClip, m_FarClip);
	
	m_InverseProjection = glm::inverse(m_Projection);
}

void Camera::RecalculateView()
{
	
	m_View = glm::translate(glm::mat4(1.0f), m_Position);
	m_View = glm::lookAt(m_Position, m_Position + m_ForwardDirection, upDirection);
	m_InverseView = glm::inverse(m_View);


	//glm::vec4 w = m_Position - m_View;

}

void async_ray_calc_func(Camera& camera, uint32_t width, uint32_t height, uint32_t thread_index)
{
	//if (thread_index % 2 != 0)
	//	return;
	float cx = thread_index % width;
	float cy = thread_index / width;

	float size_x = (float)width / camera.GetWorkingThreadsCount();
	float offset_x = size_x * cx;

	float size_y = (float)height / 1.0f;
	float offset_y = size_y * cy;

	for (uint32_t y = (uint32_t)offset_y; y < (uint32_t)(size_y + offset_y); y++)
	{
		for (uint32_t x = (uint32_t)offset_x; x < (uint32_t)(size_x + offset_x); x++)
		{
			glm::vec2 coordinator = { (float)x / (float)width,(float)y / (float)height };
			coordinator = coordinator * 2.0f - 1.0f;

			glm::vec4 target = camera.m_InverseProjection * glm::vec4(coordinator.x, coordinator.y, 1.0f, 1.0f);
			glm::vec3 rayDirection = glm::vec3(camera.m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f));
			camera.m_RayDirection[x + y * (size_t)width] = rayDirection;
		}
	}
	
}

void Camera::RecalculateRayDirection()
{

	m_RayDirection.resize((size_t)m_ViewportWidth * m_ViewportHeight);

	async_ray_calc_func(* this, m_ViewportWidth, m_ViewportHeight, 0);

	return;

	std::vector<std::thread> threads;
	for (uint32_t i = 0; i < GetWorkingThreadsCount(); i++)
	{
		threads.emplace_back(async_ray_calc_func, *this, m_ViewportWidth, m_ViewportHeight, i);
	}
	
	for (int i = 0; i < threads.size(); i++)
	{
		threads[i].join();
	}

	return;

	m_RayDirection.resize((size_t) m_ViewportWidth * m_ViewportHeight);
	
	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		for(uint32_t x = 0; x < m_ViewportWidth; x++)
		{
			glm::vec2 coordinator = { (float)x / (float)m_ViewportWidth,(float)y / (float)m_ViewportHeight };
			coordinator = coordinator * 2.0f - 1.0f;
	
			glm::vec4 target = m_InverseProjection * glm::vec4(coordinator.x, coordinator.y, 1.0f, 1.0f);
			glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f));
			m_RayDirection[x + y * (size_t) m_ViewportWidth] = rayDirection;
		}
	}

}