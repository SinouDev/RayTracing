#include "Renderer.h"

#include "glm/glm.hpp"

#include "Utils.h"
#include "Walnut/Timer.h"

#include "Camera.h"
#include "Ray.h"

#include <iostream>
#include <string>
#include <fstream>

#include <thread>
#include <vector>

#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void async_render_func(Renderer&, const std::shared_ptr<Camera>&, uint32_t, uint32_t, uint32_t);
void save_as_ppm_func(const char*, std::shared_ptr<Renderer::ImageBuffer>&);

Renderer::Renderer()
{
	Utils::Random::Init();

	m_ThreadScheduler = std::make_shared<std::vector<ThreadScheduler>>();

	ResizeThreadScheduler();
}

Renderer::~Renderer()
{
	StopRendering();
}

void Renderer::OnResize(uint32_t width, uint32_t height, RenderingCompleteCallback resizeDoneCallback)
{
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

void Renderer::Render(const std::shared_ptr<Camera>& camera)
{
	Walnut::Timer renderTime;

	m_Aspect = (float)m_ImageData->width / (float)m_ImageData->height;
	
	uint32_t i = 0;

	float size_x = (float)m_ImageData->width / m_ThreadCount;
	float size_y = (float)m_ImageData->height / m_ThreadCount;

	if (m_ThreadScheduler->size() < 1)
		return;

	for (uint32_t y = 0; y < m_ThreadCount; y++)
	{
		for (uint32_t x = 0; x < m_ThreadCount; x++)
		{
			uint32_t i = x + y * m_ThreadCount;
			
			float cx = size_x + size_x * x;
			float offset_x = size_x * x;
			
			float cy = size_y + size_y * y;
			float offset_y = size_y * y;

			uint32_t n_width = static_cast<uint32_t>(cx);
			uint32_t n_height = static_cast<uint32_t>(cy);
			n_height = static_cast<uint32_t>(n_height > (float)m_ImageData->height ? (float)m_ImageData->height : n_height);

			m_ThreadScheduler->at(i).Set(false, false, offset_x, offset_y, n_width, n_height/*, x, y*/);
		}
	}


	if (m_ThreadCount < 2)
	{
		async_render_func(*this, camera, m_ImageData->width, m_ImageData->height, i);
	}
	else
	{
		std::vector<std::thread> threads;
		for (i = 0; i < m_ThreadCount; i++)
		{

			threads.emplace_back([this, i, camera]() -> void {
				async_render_func(*this, camera, m_ImageData->width, m_ImageData->height, i);
			});
		}

		for (i = 0; i < m_ThreadCount; i++)
		{
			threads[i].join();
		}

	}

	m_RenderingTime = renderTime.ElapsedMillis();
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
	m_AsyncThreadRenderOneFlag = true;
	std::thread renderThread([this, camera]() -> void {

		m_AsyncThreadRunning = true;
		Render(camera);
		m_AsyncThreadRunning = false;
		m_AsyncThreadFlagRunning = false;

		if (m_ThreadDoneCallBack)
		{
			m_ThreadDoneCallBack();
			m_ThreadDoneCallBack = nullptr;
		}
	});

	renderThread.detach();
}

void Renderer::StartAsyncRender(const std::shared_ptr<Camera>& camera)
{
	if (m_AsyncThreadRunning)
		return;

	m_AsyncThreadFlagRunning = true;
	m_AsyncThreadRenderOneFlag = false;
	std::thread renderThread([this,  camera]() -> void {
		m_AsyncThreadRunning = true;
		do
		{
			Render(camera);
			
		} while (!m_AsyncThreadRecycleFlag && m_AsyncThreadFlagRunning);

		m_AsyncThreadRunning = false;

		if (m_ThreadDoneCallBack)
		{
			m_ThreadDoneCallBack();
			m_ThreadDoneCallBack = nullptr;
		}
	});

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

	//std::shared_ptr<ImageBuffer> copyImage = std::make_shared<ImageBuffer>(m_ImageData->width, m_ImageData->height, m_ScreenshotChannels, m_ScreenshotBuffer->Get<uint8_t*>());
	std::thread saveAsPPMThread(save_as_ppm_func, path, m_ScreenshotBuffer);
	saveAsPPMThread.detach();
}

void Renderer::SaveAsPNG(const char* path)
{
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
	return Ray::RayColor(ray, GetRayBackgroundColor(), m_HittableObjectList, 10);
}

void async_render_func(Renderer& renderer, const std::shared_ptr<Camera>& camera, uint32_t width, uint32_t height, uint32_t thread_index)
{
	do
	{
		Walnut::Timer renderTime;
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
					glm::vec4 color(0.0f);

					for (uint32_t s = 0; s < renderer.m_SamplingRate && renderer.m_AsyncThreadFlagRunning; ++s)
					{
						glm::vec2 coordinator = { ((float)x + Utils::Random::RandomDouble()) / ((float)width - 1.0f), ((float)y + Utils::Random::RandomDouble()) / ((float)height - 1.0f) };
						coordinator = coordinator * 2.0f - 1.0f;
						color += Ray::RayColor(
							camera->GetRay(coordinator),
							Renderer::GetRayBackgroundColor(),
							renderer.m_HittableObjectList,
							renderer.m_RayColorDepth) * (1.0f / renderer.m_SamplingRate);
					}
					renderer.m_ImageData->Get<uint32_t*>()[px] = Utils::Color::Vec4ToRGBA(glm::clamp(color, glm::vec4(0.0f), glm::vec4(1.0f)));

				}
			}
			p.rendering = false;
			p.completed = true;
		}

		if (!renderer.m_AsyncThreadRecycleFlag)
			break;

		bool a;
		do
		{
			a = false;
			for (Renderer::ThreadScheduler& p : *renderer.m_ThreadScheduler.get())
			{
				//p.rendering = false;
				if (!p.completed)
				{
					a = false;
					break;
				}
				a = true;
			}
		} while (!a);

		for (Renderer::ThreadScheduler& p : *renderer.m_ThreadScheduler.get())
		{
			p.rendering = false;
			p.completed = false;
		}

		if (!renderer.m_AsyncThreadRenderOneFlag)
		{
			if (renderer.m_ClearOnEachFrame && renderer.m_AsyncThreadFlagRunning)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(renderer.m_ClearDelay));
				renderer.m_ImageData->Clear();
			}
		}

		renderer.m_RenderingTime = renderTime.ElapsedMillis();
	} while (!renderer.m_AsyncThreadRenderOneFlag && renderer.m_AsyncThreadFlagRunning);
}

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