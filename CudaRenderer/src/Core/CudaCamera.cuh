#pragma once

#include "gmath.cuh"
#include "Ray.cuh"

enum CameraType : int32_t {
	Perspective_Camera = 0,
	Orthographic_Camera
};

struct CameraComponent
{
	struct CameraLens {
		union
		{
			struct {
				float m_VerticalFOV;
				float m_NearClip;
				float m_FarClip;
			};
			float cameraFiled[3];
		};
		float m_AspectRatio;
		float m_Aperture;
		float m_FocusDistance;
		float m_Time0 = 0.0f;
		float m_Time1 = 1.0f;
		float m_LensRadius = 0.0f;
		CameraType m_CameraType = Perspective_Camera;
	} m_CameraLens;

	struct CameraView {
		glm::mat2x3 m_ViewCoordMat{ 0.0f };
		glm::mat4 m_Projection{ 1.0f };
		glm::mat4 m_View{ 1.0f };
		glm::mat4 m_InverseProjection{ 1.0f };
		glm::mat4 m_InverseView{ 1.0f };
	} m_CameraView;

	struct MouseSpec {
		float m_MouseSensetivity = 0.002f;
		float m_MoveSpeed = 5.0f;
		glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };
	} m_MouseSpecs;

	struct CameraPos {
		glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };
		glm::vec3 m_ForwardDirection{ 0.0f, 0.0f, 0.0f };
	} m_CameraPos;

	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
};

class CudaRenderer;

class Camera
{
public:

	Camera(float verticalFOV = 45.0f, float nearClip = 0.1f, float farClip = 100.0f, float aspectRatio = 16.0f / 9.0f, float aperture = 1.0f, float focusDistance = 1.0f, float _time0 = 0.0f, float _time1 = 0.0f);
	Camera(const CameraComponent::CameraLens& cameraLens);
	__device__ __host__ Camera(const Camera& camera)
		: m_Component(camera.m_Component)
		{}

	bool OnUpdate(float ts);
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
	void SetCameraType(CameraType type);
	float GetRotationSpeed();

	__device__ __host__ const glm::mat4& GetProjection() const { return m_Component.m_CameraView.m_Projection; }
	__device__ __host__ const glm::mat4& GetInverseProjection() const { return m_Component.m_CameraView.m_InverseProjection; }
	__device__ __host__ const glm::mat4& GetView() const { return m_Component.m_CameraView.m_View; }
	__device__ __host__ const glm::mat4& GetInverseView() const { return m_Component.m_CameraView.m_InverseView; }
	__device__ __host__ const glm::vec3& GetPosition() const { return m_Component.m_CameraPos.m_Position; }
	__device__ __host__ const glm::vec3& GetDirection() const { return m_Component.m_CameraPos.m_ForwardDirection; }
	__device__ __host__ float GetAspectRatio() { return m_Component.m_CameraLens.m_AspectRatio; }
	__device__ __host__ float GetAperture() { return m_Component.m_CameraLens.m_Aperture; }
	__device__ __host__ float GetFocusDistance() { return m_Component.m_CameraLens.m_FocusDistance; }
	__device__ __host__ float GetLensRadius() { return m_Component.m_CameraLens.m_LensRadius; }
	inline float& GetMoveSpeed() { return m_Component.m_MouseSpecs.m_MoveSpeed; }

	__device__ __host__ const CameraComponent& GetComponent() const
	{
		return m_Component;
	}

	__device__ __host__ CameraComponent& GetComponent()
	{
		return m_Component;
	}

	void* operator new(size_t size)
	{
		void* ptr;
		cudaMallocManaged(&ptr, size);
		return ptr;
	}

	void operator delete(void* ptr)
	{
		cudaFree(ptr);
	}

protected:

	void RecalculateProjection();
	void RecalculateView();

protected:

	CameraComponent m_Component;

};