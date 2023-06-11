#include "CudaRenderer.cuh"

#include "Walnut\Timer.h"

#include "CudaReference.h"
#include "gmath.cuh"
#include "CudaColor.cuh"
#include "Scene.cuh"

#include <stdio.h>


namespace CUDA {
	using namespace SGOL;

	__global__ void InitCurand(CudaRenderer::ScreenDataBuffer screen, curandState_t* state, uint32_t seed)
	{
		//uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		//uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

		//size_t tid = screen.Index(x, y);
		for(size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < screen.Size(); tid += blockDim.x * gridDim.x)
			curand_init(clock64() * seed * tid, 0, 0, &state[tid]);
	}

	__device__ void OldStyle_RenderCudaScreen(CudaRenderer& renderer)
	{
		// update only when needed
		if (renderer.m_FrameIndex > renderer.m_AccumulationThreshold)
			return;

		const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
		const float scale = 1.0f / renderer.m_SamplingRate;
		CudaRenderer::ScreenDataBuffer& screen = renderer.ScreenBuffer();
		CudaRenderer::AccumulationDataBuffer& accumulation = renderer.m_AccumulationDataBuffer;

		
		const glm::vec2 point = glm::vec2((float) x, (float)y);
		const glm::vec2 dimention = screen.Dimentions();
		const size_t i = screen.Index(x, y);
		Color color = 0.0f;

		renderer.m_RandomState(i);

		if (renderer.m_FrameIndex == 1)
			accumulation[i] = glm::vec4(0.0f);

		glm::vec2 coord;
		uint8_t samples = 0;
		for (; samples < renderer.m_SamplingRate; samples++)
		{
			if (renderer.m_SamplingRate > 1)
				coord = (point + renderer.m_RandomState.randomVec2()) / (dimention - 1.0f);// glm::vec2(point.x + renderer.m_RandomState.random_float() / (float)(screen.Width() - 1), point.y + renderer.m_RandomState.random_float() / (float)(screen.Height() - 1));
			else
				coord = point / dimention;// glm::vec2(point.x / (float)screen.Width(), point.y / (float)screen.Height());
			coord = coord * 2.0f - 1.0f;

			color += renderer.PerPixel(coord);
		}

		color *= scale;

		if (!renderer.m_Settings.accumulate)
		{
			color.a = 1.0f;
			screen[i] = static_cast<uint32_t>(color.Clamp());
			return;
		}

		accumulation[i] += color;

		Color accumulationColor = accumulation[i];
		accumulationColor /= (float)renderer.m_FrameIndex;
		accumulationColor.Clamp();
		accumulationColor.a = 1.0f;

		screen[i] = static_cast<uint32_t>(accumulationColor);
	}

	__device__ SGOL::Color ProcessBlurEffect(const CudaRenderer::AccumulationDataBuffer& accumulation, size_t accumulationIndex, uint32_t boxSize)
	{
		// blur pre-processing algorithm
		SGOL::Color sampledColor(0.0f);

		for (uint32_t s = 0; s < boxSize; s++)
		{
			glm::vec2 coord = accumulation.Point(accumulationIndex);
			glm::vec2 sampleCoord((float)(s % boxSize), (float)(s / boxSize));

			sampleCoord -= (float)(boxSize / 2);
			glm::clamp(sampleCoord, glm::vec2(0.0f), accumulation.Dimentions());

			glm::vec2 workingCoord = coord + sampleCoord;

			sampledColor += accumulation(workingCoord);
		}

		return sampledColor / (float)boxSize;
	}

	__global__ void RendererPostProcessing(CudaRenderer renderer)
	{
		// update only when needed
		if (renderer.m_FrameIndex > renderer.m_AccumulationThreshold)
			return;

		CudaRenderer::ScreenDataBuffer& screen = renderer.m_ScreenBuffer;
		CudaRenderer::AccumulationDataBuffer& accumulation = renderer.m_AccumulationDataBuffer;
		
		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		const size_t add_i = blockDim.x * gridDim.x;
		for (; i < screen.Size(); i += add_i)
		{
			Color accumulationColor = accumulation[i];// ProcessBlurEffect(accumulation, i, renderer.m_BlurSamplingArea);;
			accumulationColor /= (float)renderer.m_FrameIndex;
			accumulationColor.Clamp();
			accumulationColor.a = 1.0f;

			screen[i] = static_cast<uint32_t>(accumulationColor);
		}
	}

	__global__ void RenderCudaScreen(CudaRenderer renderer)
	{
		// use old style renderering (pixel per thread)
		if (renderer.m_Settings.oldStyleThreading)
		{
			OldStyle_RenderCudaScreen(renderer);
			return;
		}

		// update only when needed
		if (renderer.m_FrameIndex > renderer.m_AccumulationThreshold)
			return;

		CudaRenderer::ScreenDataBuffer& screen = renderer.m_ScreenBuffer;
		CudaRenderer::AccumulationDataBuffer& accumulation = renderer.m_AccumulationDataBuffer;

		size_t i = blockIdx.x * blockDim.x + threadIdx.x;
		const size_t add_i = blockDim.x * gridDim.x;
		for (; i < screen.Size(); i += add_i)
		{
			glm::vec2 point = screen.Point(i);
			glm::vec2 dimention = screen.Dimentions();
			Color color = 0.0f;

			renderer.m_RandomState(i);

			if (renderer.m_FrameIndex == 1)
				accumulation[i] = glm::vec4(0.0f);

			glm::vec2 coord;
			uint8_t samples = 0;
			for (; samples < renderer.m_SamplingRate; samples++)
			{
				coord = (point + renderer.m_RandomState.randomVec2()) / (dimention - 1.0f);
				coord = coord * 2.0f - 1.0f;
				color = renderer.PerPixel(coord);
			}

			if (!renderer.m_Settings.accumulate)
			{
				accumulation[i] = color;
				continue;
			}
			accumulation[i] += color;
		}
	}
}

__device__ void CudaRenderer::HandleMaterialForSphere(const SceneComponent& sceneComp, const HitPayload& payload, Ray& ray)
{
	using namespace SGOL;

	const Sphere& sphere = sceneComp.spheres[payload.objectIndex];
	const Material& material = sceneComp.materials[sphere.materialIndex];

	switch (material.type)
	{
		case Material_Isotropic:
			// isotropic material
			ray.origin = payload.worldPosition;
			ray.direction = glm::normalize(m_RandomState.randomInUnitSphere() + payload.worldNormal);
			break;

		case Material_Lambertian:
			// lambertian material
			glm::vec3 scatter_direction = (payload.worldNormal - m_RandomState.randomInUnitSphere());
			if (gmath::nearZero(scatter_direction))
				scatter_direction = payload.worldNormal;
			//ray.origin = payload.worldPosition + payload.worldNormal * 0.001f;
			ray.origin = payload.worldPosition;
			ray.direction = scatter_direction;
			break;

		case Material_Metal:
			// metalic material
			glm::vec3 reflected = glm::reflect(gmath::unit_vec(ray.direction), payload.worldNormal);
			ray.origin = payload.worldPosition;
			glm::vec3 reflectionRes = reflected + material.melatic * m_RandomState.randomInUnitSphere();
			//if (glm::dot(payload.worldPosition, reflectionRes) <= 0.0f)
			//	break;
			ray.direction = reflectionRes;
			break;

		case Material_Dilectric:
			// dilectric material
			float refraction_ratio = payload.front_face ? (1.0f / material.refractionIndex) : material.refractionIndex;
			glm::vec3 unit_direction = gmath::unit_vec(ray.direction);// / Utils::Math::Q_Length(ray.GetDirection());

			float cos_theta = glm::min(glm::dot(-unit_direction, payload.worldNormal), 1.0f);
			float sin_theta = gmath::fastSqrt(1.0f - cos_theta * cos_theta);

			bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
			glm::vec3 direction;

			if (cannot_refract || gmath::reflectness(cos_theta, refraction_ratio) > m_RandomState.random_float())
				direction = glm::reflect(unit_direction, payload.worldNormal);
			
			else
				direction = gmath::refract(unit_direction, payload.worldNormal, refraction_ratio);
			
			ray.origin = payload.worldPosition;
			ray.direction = direction;
			const_cast<Material&>(material).albedo = Color(1.0f);
			break;

		case Material_Emissve:
			// light defuse material
			// nothing special to for it
	}
}

__device__ CudaRenderer::Color CudaRenderer::PerPixel(glm::vec2 coord)
{
	using namespace SGOL;
	
	const auto& m_Component = m_ActiveCamera->GetComponent();

	Ray ray = GetRayForComponent(m_Component, coord);

	Color light(0.0f);
	glm::vec3 contribution(1.0f);
	
	uint32_t bounces = 0;
	HitPayload payload;
	for (;bounces < m_ActiveScene->GetRayBouncingRate(); bounces++)
	{
		payload = TraceRay(ray);

		// if no hit break the loop and return the sky color defined in the current active scene
		if (payload.objectIndex < 0)
		{
			light += m_ActiveScene->GetSkyColor(ray) * contribution;
			break;
		}

		const auto& sceneComp = m_ActiveScene->GetSceneComponent();
		const Sphere& sphere = sceneComp.spheres[payload.objectIndex];
		const Material& material = sceneComp.materials[sphere.materialIndex];

		HandleMaterialForSphere(sceneComp, payload, ray);
		
		contribution *= material.albedo;
		light += material.emition * material.lightIntencity;
	}
	
	// avrage the sampled rays if no object was hit when the loop ends aka returned a sky color
	if(payload.objectIndex < 0)
		return light;
	return light / (float)bounces;
}

__device__ CudaRenderer::HitPayload CudaRenderer::TraceRay(const Ray& ray)
{
	using namespace SGOL;
	const auto& sceneComp = m_ActiveScene->GetSceneComponent();
	const auto& cameraComp = m_ActiveCamera->GetComponent();
	const auto& spheres = sceneComp.spheres;

	int32_t closestSphere = -1;
	float closestHit = gmath::infinity;
	for (int32_t index = 0; index < sceneComp.spheres_size; index++)
	{
		const Sphere& sphere = spheres[index];

		if (!sphere.draw)
			continue;

		glm::vec3 origin = ray.origin - sphere.position;
		float a = glm::length2(ray.direction);
		float half_b = glm::dot(origin, ray.direction);
		float c = glm::length2(origin) - sphere.radius * sphere.radius;
		float discriminant = half_b * half_b - a * c;

		if (discriminant < 0.0f)
			continue;

		float discriminant_sqrt = glm::sqrt(discriminant);

		float root = (-half_b - discriminant_sqrt) / a;

		if (root < 0.001f || root > gmath::infinity)
		{
			root = (-half_b + discriminant_sqrt) / a;
			if (root < 0.001f || root > gmath::infinity)
				continue;
		}

		if (root < closestHit)
		{
			closestSphere = index;
			closestHit = root;
		}
	}

	if (closestSphere < 0)
		return Miss(ray);

	return ClosestHit(ray, closestHit, closestSphere);
}

__device__ CudaRenderer::HitPayload CudaRenderer::ClosestHit(const Ray& ray, float hitDistance, int32_t objectIndex)
{
	using namespace SGOL;
	const auto& sceneComp = m_ActiveScene->GetSceneComponent();
	const auto& spheres = sceneComp.spheres;

	HitPayload payload;
	payload.hitDistance = hitDistance;
	payload.objectIndex = objectIndex;

	const Sphere& sphere = spheres[objectIndex];
	const Material& material = sceneComp.materials[sphere.materialIndex];

	payload.worldPosition = ray.origin + ray.direction * hitDistance;
	glm::vec3 outwardNormal = glm::normalize(payload.worldPosition - sphere.position);
	payload.set_face_normal(ray, outwardNormal);

	return payload;
}

__device__ CudaRenderer::HitPayload CudaRenderer::Miss(const Ray& ray)
{
	HitPayload payload;
	payload.objectIndex = -1;
	return payload;
}

__device__ SGOL::Ray CudaRenderer::GetRayForComponent(const CameraComponent& component, const glm::vec2& coord)
{
	using namespace SGOL;
	glm::vec3 rd = component.m_CameraLens.m_LensRadius * m_RandomState.randomInUnitDisk();
	glm::vec3 offset = component.m_CameraView.m_ViewCoordMat[0] * rd.x + component.m_CameraView.m_ViewCoordMat[1] * rd.y;
	glm::vec4 target = component.m_CameraView.m_InverseProjection * glm::vec4(coord.x, coord.y, 1.0f, 1.0f);
	glm::vec3 rayDirection = glm::vec3(component.m_CameraView.m_InverseView * glm::vec4(gmath::unit_vec(glm::vec3(target) / target.w), 0.0f)) * component.m_CameraLens.m_FocusDistance;

	return Ray(component.m_CameraPos.m_Position + offset, rayDirection - offset, m_RandomState.random_float(component.m_CameraLens.m_Time0, component.m_CameraLens.m_Time1));
}

void CudaRenderer::Render(const Scene* scene, const Camera* camera)
{
	Walnut::Timer timer;
	m_ActiveCamera = camera;
	m_ActiveScene = scene;

	//if(m_FrameIndex == 1)
	//	memset(m_AccumulationDataBuffer, 0, Width() * Height() * sizeof(AccumulationDataBuffer::Type));

	dim3 numsOfblocks = 32 * m_SumSMs;
	dim3 threadsPerBlock = 256;

	if (m_Settings.oldStyleThreading)
	{
		numsOfblocks = m_ScreenBuffer.m_NumBlocks;
		threadsPerBlock = m_ScreenBuffer.m_ThreadsPerBlock;
	}
	//numsOfblocks = 1;
	//threadsPerBlock = 1;

	CUDA::RenderCudaScreen<<<numsOfblocks, threadsPerBlock>>>(*this);
	CUDA::RendererPostProcessing<<<numsOfblocks, threadsPerBlock>>>(*this);

	cudaError_t err = cudaDeviceSynchronize();

	if (m_Settings.accumulate)
		m_FrameIndex++;
	else
		m_FrameIndex = 1;

	m_RenderTime = timer.ElapsedMillis();
}

CudaRenderer::CudaRenderer()
	: CudaRenderer(0, 0)
{
}

CudaRenderer::CudaRenderer(size_t width, size_t height)
	: m_ScreenBuffer(width, height)
{
}

__host__ __device__ CudaRenderer::CudaRenderer(CudaRenderer& renderer)
	: m_ScreenBuffer(renderer.m_ScreenBuffer), m_AccumulationDataBuffer(renderer.m_AccumulationDataBuffer), m_RandomState(renderer.m_RandomState), m_Settings(renderer.m_Settings), m_SamplingRate(renderer.m_SamplingRate), m_IsCopy(true), m_FrameIndex(renderer.m_FrameIndex), m_AccumulationThreshold(renderer.m_AccumulationThreshold), m_ActiveCamera(renderer.m_ActiveCamera), m_ActiveScene(renderer.m_ActiveScene), m_BlurSamplingArea(renderer.m_BlurSamplingArea)//, m_DeviceID(renderer.m_DeviceID), m_SumSMs(renderer.m_SumSMs)
{}

CudaRenderer::~CudaRenderer()
{
	if(!m_IsCopy)
		m_RandomState.Clean();
}

void CudaRenderer::Resize(size_t width, size_t height)
{
	m_ScreenBuffer.Resize(width, height);
	m_AccumulationDataBuffer.Resize(width, height);
	initCudaRandomStates();
	ResetFrameIndex();
}

void CudaRenderer::initCudaRandomStates()
{

	if (m_RandomState)
		m_RandomState.Clean();

	cudaMalloc(&m_RandomState, m_ScreenBuffer.Size() * sizeof(curandState_t));

	cudaDeviceGetAttribute(&m_SumSMs, cudaDevAttrMultiProcessorCount, m_DeviceID);
	CUDA::InitCurand<<<32 * m_SumSMs, 256>>>(m_ScreenBuffer, m_RandomState, 1);

	cudaError_t err = cudaDeviceSynchronize();
	
}

