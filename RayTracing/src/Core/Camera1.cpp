#include "Camera1.h"

#include "Utils.h"
#include "Ray.h"
#include "Random.h"

#include "Walnut/Input/Input.h"
#include "Walnut/Input/KeyCodes.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <thread>

constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);

Camera::Camera(float verticalFOV, float nearClip, float farClip, float aspectRatio, float aperture, float focusDistance, float _time0, float _time1)
	: m_VerticalFOV(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip), m_AspectRatio(aspectRatio), m_Aperture(aperture), m_FocusDistance(focusDistance), m_Time0(_time0), m_Time1(_time1)
{
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

	m_ViewportWidth = width;
	m_ViewportHeight = height;

	//m_AspectRatio = m_ViewportWidth * m_ViewportHeight;

	RecalculateProjection();
	RecalculateView();
	//RecalculateRayDirection();
}

void Camera::SetNearClip(float nearClip)
{
}

void Camera::SetFarClip(float farClip)
{
}

void Camera::LookAt(const glm::vec3& direction)
{
	m_ForwardDirection = direction;
	RecalculateView();
}

void Camera::LookFrom(const glm::vec3& position)
{
	m_Position = position;
	RecalculateView();
}

Ray Camera::GetRay(const glm::vec2& coord) const
{
	//glm::vec3 rayDirection = m_LowerLeftCorner + coord.s * m_Horizontal + coord.t * m_Vertical - m_Position;
	
	glm::vec3 rd = m_LensRadius * Random::RandomInUnitDisk();
	glm::vec3 offset = m_ViewCoordMat[0] * rd.x + m_ViewCoordMat[1] * rd.y;
	// // O: insert return statement here
	//glm::vec2 coordinator = { ((float)x + Random::RandomDouble()) / ((float)width - 1.0f), ((float)y + Random::RandomDouble()) / ((float)height - 1.0f) };

	glm::vec4 target = m_InverseProjection * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
	glm::vec3 rayDirection = glm::vec3(m_InverseView * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f)) * m_FocusDistance;

	return Ray(m_Position + offset, rayDirection - offset, Random::RandomDouble(m_Time0, m_Time1));
}

void Camera::SetAspectRatio(float a)
{
	if (m_AspectRatio == a)
		return;
	m_AspectRatio = a;
	RecalculateView();
}

void Camera::SetAperture(float a)
{
	if (m_Aperture == a)
		return;
	m_Aperture = a;
	RecalculateView();
}

void Camera::SetFocusDistance(float d)
{
	if (m_FocusDistance == d)
		return;
	m_FocusDistance = d;
	RecalculateView();
}

void Camera::SetLensRadius(float l) {
	if (m_LensRadius == l)
		return;
	m_LensRadius = l;
	RecalculateView();
}

void Camera::SetFOV(float fov)
{
	if (m_VerticalFOV == fov)
		return;
	m_VerticalFOV = fov;
	RecalculateView();
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
	float theta = glm::radians(m_VerticalFOV);
	float h = glm::tan(theta / 2.0f);
	//vh = 2.0f * h;
	//vw = m_AspectRatio * vh;
	
	auto center = m_Position + m_ForwardDirection;
	
	m_View = glm::translate(glm::mat4(1.0f), m_Position);
	m_View = glm::lookAt(m_Position, center, upDirection);
	m_InverseView = glm::inverse(m_View);

	

	//glm::vec3 w = Utils::UnitVec(center - m_Position);
	//m_ViewCoordMat[0] = Utils::UnitVec(glm::cross(w, up));
	//m_ViewCoordMat[1] = glm::cross(u, w);

	//glm::vec3 w = glm::vec3(m_View[0][2], m_View[1][2], m_View[2][2]);
	m_ViewCoordMat[0] = glm::vec3(m_View[0][0], m_View[1][0], m_View[2][0]);
	m_ViewCoordMat[1] = glm::vec3(m_View[0][1], m_View[1][1], m_View[2][1]);

	//m_Horizontal = vw * u * m_FocusDistance;
	//m_Vertical = vh * v * m_FocusDistance;
	//m_LowerLeftCorner = m_Position - m_Horizontal / 2.0f - m_Vertical / 2.0f + w * m_FocusDistance;

	m_LensRadius = m_Aperture / 2.0f;

}

/*
void Camera::RecalculateRayDirection()
{
	return;
	m_RayDirection.resize(m_ViewportWidth * m_ViewportHeight);
	for (uint32_t y = 0; y < m_ViewportHeight; y++)
	{
		//if (y >= height)
			//break;
		for (uint32_t x = 0; x < m_ViewportWidth; x++)
		{

			uint32_t px = x + m_ViewportWidth * y;
			m_RayDirection[px].clear();

			for(uint32_t s = 0; s < 1;s++)
			{
				glm::vec2 coordinator = { ((float)x + Random::RandomDouble()) / ((float)m_ViewportWidth - 1.0f), ((float)y + Random::RandomDouble()) / ((float)m_ViewportHeight - 1.0f) };
				m_RayDirection[px].emplace_back(m_LowerLeftCorner + coordinator.s * m_Horizontal + coordinator.t * m_Vertical - m_Position);
			}

			
		}
	}
}

void async_ray_calc_func(Camera&, uint32_t, uint32_t, uint32_t)
{
}
*/