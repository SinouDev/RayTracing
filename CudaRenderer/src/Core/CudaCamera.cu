#include "CudaCamera.cuh"

#include "Walnut/Input/Input.h"
#include "Walnut/Input/KeyCodes.h"

#include "Random.cuh"
#include "CudaRenderer.cuh"

static glm::vec3 upDirection(0.0f, 1.0f, 0.0f);

Camera::Camera(float verticalFOV, float nearClip, float farClip, float aspectRatio, float aperture, float focusDistance, float _time0, float _time1)
{
	m_Component.m_CameraLens.m_VerticalFOV = verticalFOV;
	m_Component.m_CameraLens.m_NearClip = nearClip;
	m_Component.m_CameraLens.m_FarClip = farClip;
	m_Component.m_CameraLens.m_AspectRatio = aspectRatio;
	m_Component.m_CameraLens.m_Aperture = aperture;
	m_Component.m_CameraLens.m_FocusDistance = focusDistance;
	m_Component.m_CameraLens.m_Time0 = _time0;
	m_Component.m_CameraLens.m_Time1 = _time1;
}

Camera::Camera(const CameraComponent::CameraLens& cameraLens)
{
	m_Component.m_CameraLens = cameraLens;
}

bool Camera::OnUpdate(float ts)
{
	glm::vec2 mousePosition = Walnut::Input::GetMousePosition();
	glm::vec2 delta = (mousePosition - m_Component.m_MouseSpecs.m_LastMousePosition) * m_Component.m_MouseSpecs.m_MouseSensetivity;
	m_Component.m_MouseSpecs.m_LastMousePosition = mousePosition;

	if (!Walnut::Input::IsMouseButtonDown(Walnut::MouseButton::Right))
	{
		Walnut::Input::SetCursorMode(Walnut::CursorMode::Normal);
		return false;
	}

	Walnut::Input::SetCursorMode(Walnut::CursorMode::Locked);

	bool mouseMoved = false;

	glm::vec3 rightDirection = glm::cross(m_Component.m_CameraPos.m_ForwardDirection, upDirection);

	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::W))
	{
		m_Component.m_CameraPos.m_Position += m_Component.m_CameraPos.m_ForwardDirection * m_Component.m_MouseSpecs.m_MoveSpeed * ts;
		mouseMoved = true;
	}
	else if (Walnut::Input::IsKeyDown(Walnut::KeyCode::S))
	{
		m_Component.m_CameraPos.m_Position -= m_Component.m_CameraPos.m_ForwardDirection * m_Component.m_MouseSpecs.m_MoveSpeed * ts;
		mouseMoved = true;
	}

	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::A))
	{
		m_Component.m_CameraPos.m_Position -= rightDirection * m_Component.m_MouseSpecs.m_MoveSpeed * ts;
		mouseMoved = true;
	}
	else if (Walnut::Input::IsKeyDown(Walnut::KeyCode::D))
	{
		m_Component.m_CameraPos.m_Position += rightDirection * m_Component.m_MouseSpecs.m_MoveSpeed * ts;
		mouseMoved = true;
	}

	if (Walnut::Input::IsKeyDown(Walnut::KeyCode::Q))
	{
		m_Component.m_CameraPos.m_Position -= upDirection * m_Component.m_MouseSpecs.m_MoveSpeed * ts;
		mouseMoved = true;
	}
	else if (Walnut::Input::IsKeyDown(Walnut::KeyCode::E))
	{
		m_Component.m_CameraPos.m_Position += upDirection * m_Component.m_MouseSpecs.m_MoveSpeed * ts;
		mouseMoved = true;
	}

	if (delta.x != 0.0f || delta.y != 0.0f)
	{
		float pitchDelta = delta.y * GetRotationSpeed();
		float yawDelta = delta.x * GetRotationSpeed();

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection), glm::angleAxis(-yawDelta, upDirection)));
		m_Component.m_CameraPos.m_ForwardDirection = glm::rotate(q, m_Component.m_CameraPos.m_ForwardDirection);

		mouseMoved = true;

	}

	if (mouseMoved)
	{
		RecalculateView();
		//RecalculateRayDirection();
	}

	return mouseMoved;

}

void Camera::OnResize(uint32_t width, uint32_t height)
{
	if (width == m_Component.m_ViewportWidth && height == m_Component.m_ViewportHeight)
	{
		return;
	}

	m_Component.m_ViewportWidth = width;
	m_Component.m_ViewportHeight = height;

	//m_AspectRatio = m_ViewportWidth * m_ViewportHeight;

	RecalculateProjection();
	RecalculateView();
	//RecalculateRayDirection();
}

void Camera::SetNearClip(float nearClip)
{
	if (m_Component.m_CameraLens.m_NearClip == nearClip)
		return;
	m_Component.m_CameraLens.m_NearClip = nearClip;
	RecalculateProjection();
}

void Camera::SetFarClip(float farClip)
{
	if (m_Component.m_CameraLens.m_FarClip == farClip)
		return;
	m_Component.m_CameraLens.m_FarClip = farClip;
	RecalculateProjection();
}

void Camera::LookAt(const glm::vec3& direction)
{
	m_Component.m_CameraPos.m_ForwardDirection = direction;
	RecalculateView();
}

void Camera::LookFrom(const glm::vec3& position)
{
	m_Component.m_CameraPos.m_Position = position;
	RecalculateView();
}

void Camera::SetAspectRatio(float a)
{
	if (m_Component.m_CameraLens.m_AspectRatio == a)
		return;
	m_Component.m_CameraLens.m_AspectRatio = a;
	RecalculateView();
}

void Camera::SetAperture(float a)
{
	if (m_Component.m_CameraLens.m_Aperture == a)
		return;
	m_Component.m_CameraLens.m_Aperture = a;
	RecalculateView();
}

void Camera::SetFocusDistance(float d)
{
	if (m_Component.m_CameraLens.m_FocusDistance == d)
		return;
	m_Component.m_CameraLens.m_FocusDistance = d;
	RecalculateView();
}

void Camera::SetLensRadius(float l) {
	if (m_Component.m_CameraLens.m_LensRadius == l)
		return;
	m_Component.m_CameraLens.m_LensRadius = l;
	RecalculateView();
}

void Camera::SetFOV(float fov)
{
	if (m_Component.m_CameraLens.m_VerticalFOV == fov)
		return;
	m_Component.m_CameraLens.m_VerticalFOV = fov;
	RecalculateProjection();
}

void Camera::SetCameraType(CameraType type)
{
	if (m_Component.m_CameraLens.m_CameraType == type)
		return;
	m_Component.m_CameraLens.m_CameraType = type;
	RecalculateProjection();
}

float Camera::GetRotationSpeed()
{
	return 0.3f;
}

void Camera::RecalculateProjection()
{
	switch (m_Component.m_CameraLens.m_CameraType)
	{
		case Perspective_Camera:
			m_Component.m_CameraView.m_Projection = glm::perspectiveFov(glm::radians(m_Component.m_CameraLens.m_VerticalFOV), (float)m_Component.m_ViewportWidth, (float)m_Component.m_ViewportHeight, m_Component.m_CameraLens.m_NearClip, m_Component.m_CameraLens.m_FarClip);
			break;
		case Orthographic_Camera:
			m_Component.m_CameraView.m_Projection = glm::ortho(-1.0f, 1.0f, -1.0f / m_Component.m_CameraLens.m_AspectRatio, 1.0f / m_Component.m_CameraLens.m_AspectRatio, -1.0f, 10.0f);
			break;
	}

	m_Component.m_CameraView.m_InverseProjection = glm::inverse(m_Component.m_CameraView.m_Projection);
}

void Camera::RecalculateView()
{
	//float theta = Utils::Math::Radians(m_VerticalFOV);
	//float h = Utils::Math::Tan(theta / 2.0f);
	//vh = 2.0f * h;
	//vw = m_AspectRatio * vh;

	auto center = m_Component.m_CameraPos.m_Position + m_Component.m_CameraPos.m_ForwardDirection;

	m_Component.m_CameraView.m_View = glm::translate(glm::mat4(1.0f), m_Component.m_CameraPos.m_Position);
	m_Component.m_CameraView.m_View = glm::lookAt(m_Component.m_CameraPos.m_Position, center, upDirection);
	m_Component.m_CameraView.m_InverseView = glm::inverse(m_Component.m_CameraView.m_View);



	//Vec3 w = Utils::Math::UnitVec(center - m_Position);
	//m_ViewCoordMat[0] = Utils::UnitVec(Utils::Math::Cross(w, up));
	//m_ViewCoordMat[1] = Utils::Math::Cross(u, w);

	//Vec3 w = Vec3(m_View[0][2], m_View[1][2], m_View[2][2]);
	m_Component.m_CameraView.m_ViewCoordMat[0] = glm::vec3(m_Component.m_CameraView.m_View[0][0], m_Component.m_CameraView.m_View[1][0], m_Component.m_CameraView.m_View[2][0]);
	m_Component.m_CameraView.m_ViewCoordMat[1] = glm::vec3(m_Component.m_CameraView.m_View[0][1], m_Component.m_CameraView.m_View[1][1], m_Component.m_CameraView.m_View[2][1]);

	//m_Horizontal = vw * u * m_FocusDistance;
	//m_Vertical = vh * v * m_FocusDistance;
	//m_LowerLeftCorner = m_Position - m_Horizontal / 2.0f - m_Vertical / 2.0f + w * m_FocusDistance;

	m_Component.m_CameraLens.m_LensRadius = m_Component.m_CameraLens.m_Aperture / 2.0f;

}