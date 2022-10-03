#include "Renderer.h"

#include "Utils/Utils.h"
#include "Utils/Random.h"
#include "Utils/Color.h"

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

using Utils::Math::Coord;
using Utils::Math::Color3;
using Utils::Math::Color4;

void async_render_func(Renderer&, const std::shared_ptr<Camera>&, size_t, size_t, uint32_t);
void save_as_ppm_func(const char*, Renderer::ImageBufferPtr&);

Renderer::Renderer()
{
	Utils::Random::Init();

	//m_ThreadScheduler = std::make_shared<std::vector<std::unique_ptr<ThreadScheduler>>>();

	ResizeThreadScheduler();
}

Renderer::~Renderer()
{
	StopRendering();
}

void Renderer::OnResize(uint32_t width, uint32_t height, OnResizeCallback resizeDoneCallback)
{
	if (m_ImageData.Data())
	{
		if (m_ImageData.Width() == width && m_ImageData.Height() == height)
			return;
	}
	else
	{
		//m_ImageData = std::make_shared<ImageBuffer>();
		//m_ScreenshotBuffer = std::make_shared<ImageBuffer>();
	}
	auto callback = [this, width, height, resizeDoneCallback](bool wasRendering = true)->void {
		m_ImageData.Resize(static_cast<size_t>(width), static_cast<size_t>(height));
		m_ScreenshotBuffer.Resize(static_cast<size_t>(width), static_cast<size_t>(height));
		
		if (resizeDoneCallback)
			resizeDoneCallback(wasRendering);

	};

	if (m_AsyncThreadFlagRunning)
		StopRendering(callback);
	else
		callback(false);
}

void Renderer::Render(const std::shared_ptr<Camera>& camera)
{
	Walnut::Timer renderTime;

	m_Aspect = (float)m_ImageData.Width() / (float)m_ImageData.Height();
	
	if (m_ThreadScheduler.size() < 1)
		return;
	
	int32_t i = 0;

	uint32_t a = m_ThreadCount * m_SchedulerMultiplier;

	float size_x = (float)m_ImageData.Width() / a;
	float size_y = (float)m_ImageData.Height() / a;

	//uint32_t x = 0, y = 0;
	//
	//for (;i< m_ThreadScheduler->size();i++)
	//{
	//	x = i % m_ThreadCount;
	//	y = i / m_ThreadCount;
	//
	//	float cx = size_x + size_x * x;
	//	float offset_x = size_x * x;
	//
	//	float cy = size_y + size_y * y;
	//	float offset_y = size_y * y;
	//
	//	uint32_t n_width = static_cast<uint32_t>(cx);
	//	uint32_t n_height = static_cast<uint32_t>(cy);
	//	n_height = static_cast<uint32_t>(n_height > (float)m_ImageData.Height() ? (float)m_ImageData.Height() : n_height);
	//
	//	m_ThreadScheduler->at(i).Set(false, false, offset_x, offset_y, n_width, n_height/*, x, y*/);
	//}


	for (uint32_t y = 0; y < a; y++)
	{
		for (uint32_t x = 0; x < a; x++)
		{
			uint32_t i = x + y * a;
			
			float cx = size_x + size_x * x;
			float offset_x = size_x * x;
			
			float cy = size_y + size_y * y;
			float offset_y = size_y * y;
	
			uint32_t n_width = static_cast<uint32_t>(cx);
			uint32_t n_height = static_cast<uint32_t>(cy);
			n_height = static_cast<uint32_t>(n_height > (float)m_ImageData.Height() ? (float)m_ImageData.Height() : n_height);
	
			m_ThreadScheduler.at(i)->Set(false, -1, static_cast<uint32_t>(offset_x), static_cast<uint32_t>(offset_y), n_width, n_height/*, x, y*/);
		}
	}

	if (m_ThreadCount < 2)
		async_render_func(*this, camera, m_ImageData.Width(), m_ImageData.Height(), i);
	else
	{
		std::vector<std::thread> threads;
		for (i = 0; i < m_ThreadCount; i++)
		{

			threads.emplace_back([this, i, camera]() -> void {
				async_render_func(*this, camera, m_ImageData.Width(), m_ImageData.Height(), i);
			});
		}

		for (i = 0; i < m_ThreadCount; i++)
		{
			threads[i].join();
		}

	}

	//std::cout << "Render done\n";

	m_RenderingTime = renderTime.ElapsedMillis();
}

void Renderer::ResizeThreadScheduler()
{
	auto callback = [this]() -> void {
		//m_ThreadScheduler->resize(m_ThreadCount * m_ThreadCount * m_SchedulerMultiplier);
		m_ThreadScheduler.clear();
		for (int32_t i = 0; i < m_ThreadCount * m_ThreadCount * m_SchedulerMultiplier * m_SchedulerMultiplier; i++)
		{
			m_ThreadScheduler.emplace_back(std::make_unique<ThreadScheduler>(ThreadScheduler{ false, -1, 0, 0, 0, 0 }));
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
	m_AsyncThreadFlagRunning = std::numeric_limits<size_t>().max();
	m_AsyncThreadRenderOneFlag = true;
	std::thread renderThread([this, camera]() -> void {

		m_AsyncThreadRunning = true;
		Render(camera);
		m_AsyncThreadRunning = false;
		m_AsyncThreadFlagRunning = 0x0;

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

	m_AsyncThreadFlagRunning = std::numeric_limits<size_t>().max();
	m_AsyncThreadRenderOneFlag = false;
	std::thread renderThread([this,  camera]() -> void {
		m_AsyncThreadRunning = true;
		do
		{
			Render(camera);
			
		} while (!m_AsyncThreadRecycleFlag && (bool)m_AsyncThreadFlagRunning);

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
	m_AsyncThreadFlagRunning = 0x0;
	if (!callback)
		return;
	m_ThreadDoneCallBack = callback;
}

void Renderer::SaveAsPPM(const char* path)
{

	for (uint32_t i = 0; i < m_ImageData.Size(); i++)
	{
		//uint32_t* a = ((uint32_t*)m_ScreenshotBuffer)[i];
		m_ScreenshotBuffer[i] = m_ImageData[i];
	}

	//std::shared_ptr<ImageBuffer> copyImage = std::make_shared<ImageBuffer>(m_ImageData.Width(), m_ImageData.Height(), m_ScreenshotChannels, m_ScreenshotBuffer->Get<uint8_t*>());
	std::thread saveAsPPMThread(save_as_ppm_func, path, m_ScreenshotBuffer);
	saveAsPPMThread.detach();
}

void Renderer::SaveAsPNG(const char* path)
{
#if 0
	uint32_t u = 0;
	for (int32_t y = (int32_t)m_ImageData.Height() - 1; y >= 0; --y)
	{
		for (int32_t x = 0; x < (int32_t)m_ImageData.Width(); ++x)
		{
			//uint32_t i = x + y * (int32_t)m_ImageData.Width();
			uint8_t colors[4];
			Utils::Color::RGBAtoColorChannels(colors, Utils::Color::FlipRGBA(m_ImageData(x, y)));
			for (uint32_t a = 0; a < m_ScreenshotChannels; a++)
			{
				m_ScreenshotBuffer[u++] = colors[a];
			}
		}
	}
#else
	for (size_t y = 0; y < m_ImageData.Height(); y++)
		for (size_t x = 0; x < m_ImageData.Width(); x++)
			m_ScreenshotBuffer(x, y) = m_ImageData(x, (m_ImageData.Height() - 1) - y);
#endif
	stbi_write_png(path, m_ImageData.Width<int32_t>(), m_ImageData.Height<int32_t>(), m_ScreenshotChannels, m_ScreenshotBuffer.Data<void>(), m_ImageData.Width<int32_t>() * m_ScreenshotChannels);
}

void Renderer::SetScalingEnabled(bool enable)
{
	m_ScalingEnabled = enable;
	if (!enable)
	{
		
	}
}

void Renderer::SetWorkingThreads(int32_t threads)

{
	if (threads < 0 || m_ThreadCount == threads)
		return;
	m_ThreadCount = threads;
	ResizeThreadScheduler();
}

void Renderer::SetSchedulerMultiplier(int32_t multiplier)
{
	if (multiplier < 0 || m_SchedulerMultiplier == multiplier)
		return;
	m_SchedulerMultiplier = multiplier;
	ResizeThreadScheduler();
}

Color3& Renderer::GetRayAmbientLightColorStart()
{
	static Color3 rayBackgroundColor = Utils::Color::RGBAtoVec3(0x84, 0x80, 0xFF);
	return rayBackgroundColor;
}

Color3& Renderer::GetRayAmbientLightColorEnd()
{
	static Color3 rayBackgroundColor1 = Utils::Color::RGBAtoVec3(0xFA, 0xE0, 0x95);
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
	m_ImageData.Fill(0);
	m_ScreenshotBuffer.Fill(0);
}

Color4 Renderer::RayTrace(Ray& ray)
{
	return Ray::RayColor(ray, GetRayAmbientLightColorStart(), m_HittableObjectList, 10);
}

void async_render_func(Renderer& renderer, const std::shared_ptr<Camera>& camera, size_t width, size_t height, int32_t thread_index)
{
	do
	{
		Walnut::Timer renderTime;
		for (const auto& p : renderer.m_ThreadScheduler)
		{
			if (p->rendering_thread < 0)
				p->rendering_thread = thread_index;
			else if (p->completed)
				continue;

			if (p->rendering_thread != thread_index)
				continue;

			for (uint32_t y = p->offset_y; y < (p->n_height & renderer.m_AsyncThreadFlagRunning); y++)
			{
				for (uint32_t x = p->offset_x; x < (p->n_width & renderer.m_AsyncThreadFlagRunning); x++)
				{

					//uint32_t px = x + width * y;
					Color4 color(0.0f);

					if (Ray::SimpleRayMode())
					{
						Coord coordinator = { (float)x / (float)width, (float)y / (float)height };
						coordinator = coordinator * 2.0f - 1.0f;
						color = Ray::RayColor(
							camera->GetRay(coordinator),
							Renderer::GetRayAmbientLightColorStart(),
							renderer.m_HittableObjectList, 1);
					}
					else
					{
						for (uint32_t s = 0; s < (renderer.m_SamplingRate & renderer.m_AsyncThreadFlagRunning); ++s)
						{
							Coord coordinator = { ((float)x + Utils::Random::RandomDouble()) / ((float)width - 1.0f), ((float)y + Utils::Random::RandomDouble()) / ((float)height - 1.0f) };
							coordinator = coordinator * 2.0f - 1.0f;
							color += Ray::RayColor(
								camera->GetRay(coordinator),
								Renderer::GetRayAmbientLightColorStart(),
								renderer.m_HittableObjectList,
								renderer.m_RayColorDepth) * (1.0f / renderer.m_SamplingRate);
						}
					}

					
					renderer.m_ImageData(x, y) = Utils::Color::Vec4ToRGBA(Utils::Math::Clamp(color, Color4(0.0f), Color4(1.0f)));

				}
			}
			p->completed = true;
		}

		break;
#if 0
		// not tested yet
		if (!renderer.m_AsyncThreadRecycleFlag)
			break;

		bool a;
		do
		{
			a = false;
			for (const auto& p : renderer.m_ThreadScheduler)
			{
				//p->rendering = false;
				if (!p->completed)
				{
					a = false;
					break;
				}
				a = true;
			}
		} while (!a);

		for (const auto& p : renderer.m_ThreadScheduler)
		{
			p->rendering_thread = -1;
			p->completed = false;
		}

		if (!renderer.m_AsyncThreadRenderOneFlag)
		{
			if (renderer.m_ClearOnEachFrame && renderer.m_AsyncThreadFlagRunning)
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(renderer.m_ClearDelay));
				renderer.m_ImageData.Fill(0);
			}
		}

		renderer.m_RenderingTime = renderTime.ElapsedMillis();
#endif
	} while (!renderer.m_AsyncThreadRenderOneFlag && renderer.m_AsyncThreadFlagRunning);
}

void save_as_ppm_func(const char* path, Renderer::ImageBufferPtr& image)
{

	if (!image.Data())
		return;

	//std::string content;

	std::ofstream ppmFile;
	ppmFile.open(path, std::ios_base::trunc);

	//content.append("P3\n").append(std::to_string(m_FinalImage->GetWidth())).append(" ").append(std::to_string(m_FinalImage->GetHeight())).append("\u255\n");
	ppmFile << "P3\n" << std::to_string(image.Width()) << ' ' << std::to_string(image.Height()) << "\n255\n";

	for (int32_t j = (int32_t)image.Height() - 1; j >= 0; --j)
	{
		for (int32_t i = 0; i < (int32_t)image.Width(); ++i)
		{
			//uint32_t index = i + image.Width() * j;
			Color3 colors;
			Utils::Color::RGBAtoVec3(colors, image(i, j));
			//content.append(std::to_string(static_cast<uint8_t>(colors.r * 255.0f))).append(" ").append(std::to_string(static_cast<uint8_t>(colors.g * 255.0f))).append(std::to_string(static_cast<uint8_t>(colors.b * 255.0f))).append("\n");
			ppmFile << std::to_string(static_cast<uint8_t>(colors.r * 255.0f)) << ' ' << std::to_string(static_cast<uint8_t>(colors.g * 255.0f)) << ' ' << std::to_string(static_cast<uint8_t>(colors.b * 255.0f)) << "\n";
		}
	}

	ppmFile.flush();

	ppmFile.close();
}