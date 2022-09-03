#include "Renderer.h"

#include "glm/glm.hpp"

#include "Utils.h"
#include "Random.h"
#include "Material/Lambertian.h"
#include "Material/Metal.h"
#include "Material/Dielectric.h"
#include "Object/MovingSphere.h"

#include "Camera1.h"
#include "Ray.h"

#include <iostream>
#include <string>
#include <fstream>

#include <thread>
#include <vector>

#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//struct Renderer::ImageBuffer {
//	uint32_t width, height;
//	uint32_t* buffer;
//
//	ImageBuffer(uint32_t w, uint32_t h, uint32_t* b)
//		: width(w), height(h), buffer(b)
//	{}
//
//};

void save_as_ppm_func(const char* path, std::shared_ptr<Renderer::ImageBuffer>& image)
{

	if (!image.get()) return;

	//std::string content;

	std::ofstream ppmFile;
	ppmFile.open(path, std::ios_base::trunc);

	//content.append("P3\n").append(std::to_string(m_FinalImage->GetWidth())).append(" ").append(std::to_string(m_FinalImage->GetHeight())).append("\u255\n");
	ppmFile << "P3\n" << std::to_string(image->width) << ' ' << std::to_string(image->height) << "\n255\n";

	for (int32_t j = (int32_t)image->height - 1; j >= 0; --j)
	{
		for (int32_t i = 0; i < (int32_t)image->width; ++i)
		{
			uint32_t index = i + image->width * j;
			glm::vec3 colors;
			Utils::Color::RGBAtoVec3(colors, image->buffer[index]);
			//content.append(std::to_string(static_cast<uint8_t>(colors.r * 255.0f))).append(" ").append(std::to_string(static_cast<uint8_t>(colors.g * 255.0f))).append(std::to_string(static_cast<uint8_t>(colors.b * 255.0f))).append("\n");
			ppmFile << std::to_string(static_cast<uint8_t>(colors.r * 255.0f)) << ' ' << std::to_string(static_cast<uint8_t>(colors.g * 255.0f)) << ' ' << std::to_string(static_cast<uint8_t>(colors.b * 255.0f)) << "\n";
		}
	}

	ppmFile.flush();

	ppmFile.close();
}

std::shared_ptr<Sphere> sphere6;

void p(Renderer& r)
{
	using MaterialPtr = Renderer::MaterialPtr;
	MaterialPtr ground_material = std::make_shared<Lambertian>(glm::vec3(0.5f, 0.5f, 0.5f));
	r.m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material));


	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			float choose_mat = Random::RandomDouble();
			glm::vec3 center(a + 0.9f * Random::RandomDouble(), 0.2f, b + 0.9f * Random::RandomDouble());

			if ((center - glm::vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
				std::shared_ptr<Material> sphere_material;

				if (choose_mat < 0.7f) {
					// diffuse
					auto albedo = Random::RandomVec3() * Random::RandomVec3();
					sphere_material = std::make_shared<Lambertian>(albedo);
					auto center2 = center + glm::vec3(0.0f, Random::RandomDouble(0.0f, 0.5f), 0.0f);
					r.m_HittableObjectList.Add(std::make_shared<MovingSphere>(
						center, center2, 0.0f, 1.0f, 0.2f, sphere_material));

					//r.m_HittableObjectList.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
				}
				else if (choose_mat < 0.85f) {
					// metal
					auto albedo = Random::RandomVec3(0.5f, 1.0f);
					float fuzz = Random::RandomDouble(0.0f, 0.5f);
					sphere_material = std::make_shared<Metal>(albedo, fuzz);
					r.m_HittableObjectList.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
				}
				else {
					// glass
					sphere_material = std::make_shared<Dielectric>(1.5f);
					r.m_HittableObjectList.Add(std::make_shared<Sphere>(center, 0.2f, sphere_material));
				}
			}
		}
	}

	MaterialPtr material1 = std::make_shared<Dielectric>(1.5f);
	r.m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f, material1));

	MaterialPtr material2 = std::make_shared<Lambertian>(glm::vec3(0.4f, 0.2f, 0.1f));
	r.m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

	MaterialPtr material3 = std::make_shared<Metal>(glm::vec3(0.7f, 0.6f, 0.5f), 0.0f);
	r.m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(4.0f, 1.0f, 0.0f), 1.0f, material3));
}

Renderer::Renderer()
{

	Random::Init();
	m_ThreadScheduler = std::make_shared<std::vector<ThreadScheduler>>();
	ResizeThreadScheduler();


	//MaterialPtr ground_material = std::make_shared<Lambertian>(glm::vec3(0.5, 0.5, 0.5));
	//m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(0, -1000, 0), 1000, ground_material));
	//
	//MaterialPtr material1 = std::make_shared<Dielectric>(1.5);
	//m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(0, 1, 0), 1.0, material1));
	//
	//MaterialPtr material2 = std::make_shared<Lambertian>(glm::vec3(0.4, 0.2, 0.1));
	//m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(-4, 1, 0), 1.0, material2));
	//
	//MaterialPtr material3 = std::make_shared<Metal>(glm::vec3(0.7, 0.6, 0.5), 0.0);
	//m_HittableObjectList.Add(std::make_shared<Sphere>(glm::vec3(4, 1, 0), 1.0, material3));

	p(*this);

	back_shpere = std::make_shared<Metal>(glm::vec3(0.5f, 0.5f, 0.5f), 0.15f);
	center_sphere = std::make_shared<Lambertian>(glm::vec3(0.7f, 0.3f, 0.3f));
	left_sphere = std::make_shared<Metal>(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f);
	right_sphere = std::make_shared<Metal>(glm::vec3(0.1f, 0.95f, 0.82f), 1.0f);
	small_sphere = std::make_shared<ShinyMetal>(glm::vec3(1.0f, 0.6f, 0.0f));

 	glass_sphere = std::make_shared<Dielectric>(1.019f);

	
	SpherePtr sphere1 = std::make_shared<Sphere>(glm::vec3(0.0f, -100.5f, -1.0f), 100.0f, back_shpere);
	SpherePtr sphere2 = std::make_shared<Sphere>(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, center_sphere);
	SpherePtr sphere3 = std::make_shared<Sphere>(glm::vec3(-1.0f, 0.0f, -1.0f), 0.5f, left_sphere);
	SpherePtr sphere4 = std::make_shared<Sphere>(glm::vec3(1.0f, 0.0f, -1.0f), 0.5f, right_sphere);
	SpherePtr sphere5 = std::make_shared<Sphere>(glm::vec3(0.0f, -0.35f, 1.0f), 0.15f, small_sphere);

	sphere6 = std::make_shared<Sphere>(glm::vec3(0.0f, -0.35f, 1.0f), 0.15f, small_sphere);


	m_GlassSphere = std::make_shared<Sphere>(glm::vec3(0.0f, 0.0f, 1.0f), -0.5f, glass_sphere);

	return;
	

	m_HittableObjectList.Add(sphere1);
	m_HittableObjectList.Add(sphere2);
	m_HittableObjectList.Add(sphere3);
	m_HittableObjectList.Add(sphere4);
	m_HittableObjectList.Add(sphere5);
	m_HittableObjectList.Add(sphere6);
	m_HittableObjectList.Add(m_GlassSphere);

	


}

Renderer::~Renderer()
{
	StopRendering();
}

void Renderer::OnResize(uint32_t width, uint32_t height, RenderingCompleteCallback resizeDoneCallback)
{
	//if (m_FinalImage)
	//{
	//	
	//	if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
	//		return;
	//	m_FinalImage->Resize(width, height);
	//}
	//else
	//{
	//	m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	//}
	//
	//
	//delete[] m_ImageData;
	//delete[] m_ScreenshotBuffer;
	//
	//m_ImageData = new uint32_t[width * height];
	//m_ScreenshotBuffer = new uint8_t[width * height * m_ScreenshotChannels];
	if (m_ImageData)
	{
		if (m_ImageData->width == width && m_ImageData->height == height)
			return;
	}
	else
	{
		m_ImageData = std::make_shared<ImageBuffer>();
		m_ScreenshotBuffer = std::make_shared<ImageBuffer>();
	}
	auto callback = [this, width, height, resizeDoneCallback](bool wasRendering = true)->void {
		m_ImageData->Resize(width, height, 4);
		m_ScreenshotBuffer->Resize(width, height, m_ScreenshotChannels);
		
		if (resizeDoneCallback && wasRendering)
			resizeDoneCallback();

	};

	if (m_AsyncThreadFlagRunning)
		StopRendering(callback);
	else
		callback(false);

}

#define THREAD 11

void async_render_func(Renderer& renderer, Camera& camera, uint32_t width, uint32_t height, uint32_t thread_index)
{

	for (Renderer::ThreadScheduler& p : *renderer.m_ThreadScheduler.get())
	{
		if (p.completed || p.rendering)
			continue;
		p.rendering = true;
		for (uint32_t y = static_cast<uint32_t>(p.offset_y); y < p.n_height && renderer.m_AsyncThreadFlagRunning; y++)
		{
			for (uint32_t x = static_cast<uint32_t>(p.offset_x); x < p.n_width && renderer.m_AsyncThreadFlagRunning; x++)
			{

				uint32_t px = x + width * y;
				glm::vec3 color(0.0f);

				for (uint32_t s = 0; s < renderer.m_SamplingRate && renderer.m_AsyncThreadFlagRunning; ++s)
				{
					glm::vec2 coordinator = { ((float)x + Random::RandomDouble()) / ((float)width - 1.0f), ((float)y + Random::RandomDouble()) / ((float)height - 1.0f) };
					coordinator = coordinator * 2.0f - 1.0f;
					color += Ray::RayColor(camera.GetRay(coordinator), renderer.m_HittableObjectList, renderer.m_RayColorDepth) * (1.0f / renderer.m_SamplingRate);
				}

				renderer.m_ImageData->Get<uint32_t*>()[px] = Utils::Color::Vec3ToRGBA(color);
			}
		}
		p.completed = true;
	}

	//if (thread_index % 2 != 0)
	//	return;
	//
	//Ray ray(camera.GetPosition());
	//ray.GetLightDir() = renderer.m_LightDir;
	//
	//for (uint32_t t = 0; t < renderer.m_ThreadCount && renderer.m_AsyncThreadFlagRunning; t++)
	//{
	//	//if (t % 2 != 0)
	//	//	continue;
	//	float cx = thread_index % width;
	//	float cy = thread_index / width;
	//
	//	float size_x = (float)width / renderer.m_ThreadCount;
	//	float offset_x = size_x * cx;
	//
	//	float size_y = (float)height / renderer.m_ThreadCount;
	//	float offset_y = size_y * cy + size_y * t;
	//
	//	//uint32_t x_offset = offset / width;
	//	//uint32_t y_offset = offset % width;
	//	// 
	//	//std::cout << "size " << size << " / offset " << offset << "\n";
	//
	//	uint32_t n_width = static_cast<uint32_t>(size_x + offset_x);
	//	uint32_t n_height = static_cast<uint32_t>(size_y + offset_y);
	//	n_height = n_height > height ? height : n_height;
	//
	//	for (uint32_t y = static_cast<uint32_t>(offset_y); y < n_height && renderer.m_AsyncThreadFlagRunning; y++)
	//	{
	//		//if (y >= height)
				//break;
	//		for (uint32_t x = static_cast<uint32_t>(offset_x); x < n_width && renderer.m_AsyncThreadFlagRunning; x++)
	//		{
	//
	//			uint32_t px = x + width * y;
	//			//if (x >= width)
				//	break;
				//
				//glm::vec2 coord = { x / (float)width, y / (float)height };
				//
				//coord.x = coord.x * renderer.m_Aspect;
				//coord = coord * 2.0f - 1.0f;
	//
	//			glm::vec3 color(0.0f);
	//
	//			//const Camera::DirectionSamples& samples_direction = camera.GetRayDirections()[px];
				//
				//for (const auto& s_dir : camera.GetRayDirections()[px])
				//{
				//	ray.GetDirection() = s_dir;
				//	color += Ray::RayColor(ray, renderer.m_HittableObjectList, renderer.m_RayColorDepth) * (1.0f / 1), glm::vec3(0.0f), glm::vec3(1.0f);
				//}
	//
	//
	//			for (uint32_t s = 0; s < renderer.m_SamplingRate; ++s)
	//			{
	//
	//
	//				//glm::vec2 coordinator = { x / (float)width, y / (float)height };
	//				glm::vec2 coordinator = { ((float)x + Random::RandomDouble()) / ((float)width - 1.0f), ((float)y + Random::RandomDouble()) / ((float)height - 1.0f) };
	//				//coordinator = coordinator * 2.0f - 1.0f;
					//float u = (x + renderer.m_Random.Float()) / (width - 1.0f);
					//float v = (y + renderer.m_Random.Float()) / (height - 1.0f);
					//
					//glm::vec4 target = camera.GetInverseProjection() * glm::vec4(coordinator.x, coordinator.y, 1.0f, 1.0f);
					//glm::vec3 rayDirection = glm::vec3(camera.GetInverseView() * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.0f));
					//
					//glm::vec2 randomPixel = { (float)x + Random::RandomDouble(), (float)y + Random::RandomDouble() };
					//uint32_t rx = coordinator.x * width + coordinator.y * width * height;
					//
					//if (rx >= width * height) continue;
					//
					//ray.GetDirection() = camera.GetRayDirections()[coordinator.x * ((float)width - 1.0f) + coordinator.y * ((float)width - 1.0f) * ((float)height - 1.0f)];
					//ray.GetDirection() = rayDirection;
	//				coordinator = coordinator * 2.0f - 1.0f;
	//				color += Ray::RayColor(camera.GetRay(coordinator), renderer.m_HittableObjectList, renderer.m_RayColorDepth) * (1.0f / renderer.m_SamplingRate);
	//			}
	//
	//			//glm::vec3 color1(0.0f);
				//for (uint32_t s = 0; s < 0; s++)
				//{
				//	float rw = Random::RandomDouble(-1.0, 1.0f);
				//	float rh = Random::RandomDouble(-1.0, 1.0f);
				//
				//	float dx = ((rw) * 3 * width) - 0.5f;
				//	float dy = ((rh) * 3 * width) - 0.5f;
				//
				//	glm::vec3 p = glm::vec3(0.0f, 0.0f, 0.0f) + focusPoint * ray.GetDirection();
				//	glm::vec3 dir = p - glm::vec3(dx, dy, 0.0f);
				//
				//	Ray ray1(glm::vec3(dx, dy, 0.0f), dir);
				//	ray1.GetDirection() = camera.GetDirection() * ray1.GetDirection();
				//
				//	color1 += glm::clamp(Ray::RayColor(ray1, renderer.m_HittableObjectList, renderer.m_RayColorDepth) * (1.0f / renderer.m_SamplingRate), glm::vec3(0.0f), glm::vec3(1.0f));
				//
				//}
				//
				//color1 /= 10.0f;
				//
				//color += color1;
				// 
				//uint32_t px = x + width * y;
				// 
				//if (renderer.m_SamplingRate < 1)
				//{
				//	ray.GetDirection() = camera.GetRayDirections()[px];
				//	color = (Ray::RayColor(ray, renderer.m_HittableObjectList, renderer.m_RayColorDepth));
				//}
				// 
				//ray.GetDirection() = camera.GetRayDirections()[px];
				//
				//glm::vec4 color1 = renderer.RayTrace(ray);
	//
	//			renderer.m_ImageData->Get<uint32_t*>()[px] = Utils::Color::Vec3ToRGBA(color);// ColorUtils::Vec4ToRGBA(ColorUtils::Vec4ToRGBABlendColor(, color1));
	//		}
	//	}
	//}
}

void Renderer::Render(Camera& camera)
{
	m_Aspect = (float)m_ImageData->width / (float)m_ImageData->height;

	sphere6->SetCenter(glm::vec3(glm::vec2(camera.GetPosition()), camera.GetPosition().z + 0.16f));
	

	//std::thread workThread(async_render_func, *this, camera, m_FinalImage->GetWidth(), m_FinalImage->GetHeight(), 0);
	//workThread.join();
	uint32_t i = 0;

	//m_ThreadPart->resize(GetThreadCount() * GetThreadCount());

	float size_x = (float)m_ImageData->width / m_ThreadCount;
	float size_y = (float)m_ImageData->height / m_ThreadCount;

	if (m_ThreadScheduler->size() < 1)
		return;

	for (uint32_t y = 0; y < m_ThreadCount; y++)
	{
		for (uint32_t x = 0; x < m_ThreadCount; x++)
		{
			//float cx = i % m_ImageData->width;
			//float cy = i / m_ImageData->width;

			uint32_t i = x + y * m_ThreadCount;
			
			float cx = size_x + size_x * x;
			float offset_x = size_x * x;

			
			float cy = size_y + size_y * y;
			float offset_y = size_y * y;

			//uint32_t x_offset = offset / width;
			//uint32_t y_offset = offset % width;
			// 
			//std::cout << "size " << size << " / offset " << offset << "\n";

			uint32_t n_width = cx;// static_cast<uint32_t>(size_x + offset_x);
			uint32_t n_height = cy;// static_cast<uint32_t>(size_y + offset_y);
			n_height = n_height > (float)m_ImageData->height ? (float)m_ImageData->height : n_height;


			m_ThreadScheduler->at(i).Set(false, false, offset_x, offset_y, n_width, n_height, x, y);
		}
	}


	if (m_ThreadCount < 2)
	{
		async_render_func(*this, camera, m_ImageData->width, m_ImageData->height, i);
	}
	else
	{
		std::vector<std::thread> threads;
		for (i = 0; i < m_ThreadCount; i++) {

			threads.emplace_back([this, i](Camera& camera) -> void {
				async_render_func(*this, camera, m_ImageData->width, m_ImageData->height, i);
			}, camera);
			//threads[i].join();
		}

		for (int i = 0; i < GetThreadCount(); i++)
		{
			//threads[i].detach();
			threads[i].join();
		}

	}


	//threads[THREAD - 1].join();

	//std::cin.get();
	
	//m_FinalImage->SetData(m_ImageData);
}

void Renderer::Render(const std::shared_ptr<Camera>& camera)
{
	Render(*camera);
}

void Renderer::ResizeThreadScheduler()
{
	auto callback = [this]() -> void {
		m_ThreadScheduler->resize(m_ThreadCount * m_ThreadCount);
		for (uint32_t i = 0; i < m_ThreadCount * m_ThreadCount; i++)
		{
			m_ThreadScheduler->emplace_back(ThreadScheduler{ false, false, 0.0f, 0.0f, 0, 0 });
		}
	};
	if (m_AsyncThreadFlagRunning)
		StopRendering(callback);
	else
		callback();
}

void Renderer::RenderOnce(const std::shared_ptr<Camera>& camera)
{
	ClearScene();
	m_AsyncThreadFlagRunning = true;
	std::thread renderThread([this](const std::shared_ptr<Camera>& camera) -> void {

		m_AsyncThreadRunning = true;
		Render(*camera);
		m_AsyncThreadRunning = false;
		m_AsyncThreadFlagRunning = false;

		if (m_ThreadDoneCallBack)
		{
			m_ThreadDoneCallBack();
			m_ThreadDoneCallBack = nullptr;
		}
	}, camera);

	renderThread.detach();
}

void Renderer::StartAsyncRender(const std::shared_ptr<Camera>& camera)
{
	if (m_AsyncThreadRunning)
		return;

	m_AsyncThreadFlagRunning = true;
	std::thread renderThread([this](const std::shared_ptr<Camera>& camera) -> void {
		m_AsyncThreadRunning = true;
		while (m_AsyncThreadFlagRunning)
		{
			Render(*camera);
			if (m_ClearOnEachFrame && m_AsyncThreadFlagRunning)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(m_ClearDelay));
				m_ImageData->Clear();
			}
		}

		m_AsyncThreadRunning = false;

		if (m_ThreadDoneCallBack)
		{
			//void (*)(void);
			//RenderingCompleteCallback callBack = m_ThreadDoneCallBack;
			//callBack();
			m_ThreadDoneCallBack();
			//m_ThreadDoneCallBack();
			m_ThreadDoneCallBack = nullptr;
		}
	}, camera);

	renderThread.detach();
}

void Renderer::StopRendering(RenderingCompleteCallback callback)
{
	if (!m_AsyncThreadFlagRunning)
		return;
	m_AsyncThreadFlagRunning = false;
	if (!callback)
		return;
	m_ThreadDoneCallBack = callback;
}

void Renderer::SaveAsPPM(const char* path)
{

	for (uint32_t i = 0; i < m_ImageData->width * m_ImageData->height; i++)
	{
		//uint32_t* a = ((uint32_t*)m_ScreenshotBuffer)[i];
		((uint32_t*)m_ScreenshotBuffer.get())[i] = ((uint32_t*)m_ImageData.get())[i];
	}

	std::shared_ptr<ImageBuffer> copyImage = std::make_shared<ImageBuffer>(m_ImageData->width, m_ImageData->height, m_ScreenshotChannels, m_ScreenshotBuffer->Get<uint8_t*>());
	//copyImage->SetData(m_ScreenshotBuffer);
	std::thread saveAsPPMThread(save_as_ppm_func, path, copyImage);
	saveAsPPMThread.detach();
}

void Renderer::SaveAsPNG(const char* path)
{

	//uint32_t ss = 0xAE050780;
	//uint32_t dd = ColorUtils::RGBAtoBRGA(ss);

	uint32_t u = 0;
	for (int32_t y = (int32_t)m_ImageData->height - 1; y >= 0; --y)
	{
		for (int32_t x = 0; x < (int32_t)m_ImageData->width; ++x)
		{
			uint32_t i = x + y * (int32_t)m_ImageData->width;
			uint8_t colors[4];
			Utils::Color::RGBAtoColorChannels(colors, Utils::Color::FlipRGBA(m_ImageData->Get<uint32_t*>()[i]));
			for (uint32_t a = 0; a < m_ScreenshotChannels; a++)
			{
				m_ScreenshotBuffer->Get<uint8_t*>()[u++] = colors[a];
			}
		}
	}

	stbi_write_png(path, m_ImageData->width, m_ImageData->height, m_ScreenshotChannels, m_ScreenshotBuffer->Get<uint8_t*>(), m_ImageData->width * m_ScreenshotChannels);

	//std::shared_ptr<ImageBuffer> copyImage = std::make_shared<ImageBuffer>(m_FinalImage->GetWidth(), m_FinalImage->GetHeight(), m_ScreenshotBuffer);
	//copyImage->SetData(m_ScreenshotBuffer);
	//std::thread saveAsPPMThread(save_as_ppm_func, path, copyImage);
	//saveAsPPMThread.detach();
}

void Renderer::SetScalingEnabled(bool enable)
{
	m_ScalingEnabled = enable;
	if (!enable)
	{
		
	}
}

void Renderer::SetWorkingThreads(uint32_t threads)

{
	if (m_ThreadCount == threads)
		return;
	m_ThreadCount = threads;
	ResizeThreadScheduler();
}

glm::vec3& Renderer::GetRayBackgroundColor()
{
	static glm::vec3 rayBackgroundColor = Utils::Color::RGBAtoVec3(0x84, 0x80, 0xFF);
	return rayBackgroundColor;
}

glm::vec3& Renderer::GetRayBackgroundColor1()
{
	static glm::vec3 rayBackgroundColor1 = Utils::Color::RGBAtoVec3(0xFA, 0xE0, 0x95);
	return rayBackgroundColor1;
}

void Renderer::SetClearOnEachFrame(bool clear)
{
	m_ClearOnEachFrame = clear;
}

void Renderer::SetClearDelay(uint64_t ms)

{
	m_ClearDelay = 0L;
	if (m_ClearOnEachFrame)
		m_ClearDelay = ms;
}

void Renderer::ClearScene()
{
	m_ImageData->Clear();
	m_ScreenshotBuffer->Clear();
}

glm::vec4 Renderer::RayTrace(Ray& ray)
{
	return glm::vec4(Ray::RayColor(ray, m_HittableObjectList, 10), 1.0f);
}
