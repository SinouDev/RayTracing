#include "Renderer.h"

#include "glm/glm.hpp"

#include "ColorUtils.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <thread>
#include <vector>

struct Renderer::ImageBuffer {

	uint32_t width, height;
	uint32_t* buffer;
	const char* path;

	ImageBuffer() = default;

	ImageBuffer(uint32_t w, uint32_t h, uint32_t* b, const char* p)
		: width(w), height(h), buffer(b), path(p)
	{}

};

void load_from_ppm_func(Renderer* renderer, const char* path)
{

	//if (!image.get()) return;

	//std::string content;

	renderer->m_LoadedPPM = std::make_shared<Renderer::ImageBuffer>();

	renderer->m_LoadedPPM->path = path;

	std::ifstream ppmFile;
	ppmFile.open(path);

	size_t bufferSize = 0U;
	size_t channelSize = 0U;
	uint32_t bufferIndex = 0;
	//uint32_t width, height;

	//uint32_t* data = (uint32_t*) image->buffer;
	//uint32_t width = 0, height = 0;

	//content.append("P3\n").append(std::to_string(m_FinalImage->GetWidth())).append(" ").append(std::to_string(m_FinalImage->GetHeight())).append("\u255\n");
	//ppmFile << "P3\n" << std::to_string(image->width) << ' ' << std::to_string(image->height) << "\n255\n";
	if (ppmFile.is_open())
	{
		bool ppmCheck = false;
		bool whCheck = false;
		while (!ppmCheck)
		{
			std::string out;
			std::getline(ppmFile, out);

			if (out[0] == '#')
				continue;

			if (out == "P3")
			{
				ppmCheck = true;
			}
			else
				break;
		}

		while(ppmFile)
		{
			std::string out;
			std::getline(ppmFile, out);

			if (out[0] == '#')
				continue;

			if (!whCheck)
			{

				std::vector<std::string> info;

				uint8_t lastI = 0;
				out.shrink_to_fit();
				for (uint8_t i = 0; i <= out.length(); i++)
				{
					//if ()
					//{
					//	info.emplace_back(out.substr(lastI, i - lastI));
					//}
					if (i == out.length() || out[i] == ' ')
					{
						info.emplace_back(out.substr(lastI, i - lastI));
						lastI = i + 1;
					}
				}

				std::string width_string = info[0];
				std::string height_string = info[1];

				renderer->m_LoadedPPM->width = std::stoi(width_string);
				renderer->m_LoadedPPM->height = std::stoi(height_string);

				//int width_value, height_value;
				//std::istringstream (width) >> width_value;
				//std::istringstream (height) >> height_value;

				std::getline(ppmFile, out);

				channelSize = std::stoi(out);

				// TODO: make an image class instance

				bufferSize = renderer->m_LoadedPPM->width * renderer->m_LoadedPPM->height;

				renderer->m_LoadedPPM->buffer = new uint32_t[bufferSize];
				
				whCheck = true;

				continue;

			}

			//std::getline(ppmFile, out);

			glm::vec3 pixel;

			uint8_t channelCount = 0;
			uint8_t lastI = 0;
			out.shrink_to_fit();
			for (uint8_t i = 0; i <= out.length(); i++)
			{
				//if ()
				//{
				//	info.emplace_back(out.substr(lastI, i - lastI));
				//}
				if (i == out.length() || out[i] == ' ')
				{
					pixel[channelCount++] = (float)std::stoi(out.substr(lastI, i - lastI)) / (float)channelSize;
					lastI = i + 1;
				}
			}
			
			renderer->m_LoadedPPM->buffer[bufferIndex++] = ColorUtils::Vec3ToRGBA(pixel);

			if (bufferIndex >= bufferSize)
				break;
			

			//std::cout << out << '\n';
		}

		//renderer->m_LoadedPPM->imageBuffer = std::make_shared<Walnut::Image>(renderer->m_LoadedPPM->width, renderer->m_LoadedPPM->height, Walnut::ImageFormat::RGBA, renderer->m_LoadedPPM->buffer);
		
	}

	//return;
	//
	//for (int32_t j = image->height - 1; j >= 0; --j)
	//{
	//	for (int32_t i = 0; i < image->width; ++i)
	//	{
	//		uint32_t index = i + image->width * j;
	//		glm::vec3 colors;
	//		ColorUtils::RGBAtoVec3(colors, image->buffer->GetData()[index]);
	//		//content.append(std::to_string(static_cast<uint8_t>(colors.r * 255.0f))).append(" ").append(std::to_string(static_cast<uint8_t>(colors.g * 255.0f))).append(std::to_string(static_cast<uint8_t>(colors.b * 255.0f))).append("\n");
	//		//ppmFile << std::to_string(static_cast<uint8_t>(colors.r * 255.0f)) << ' ' << std::to_string(static_cast<uint8_t>(colors.g * 255.0f)) << ' ' << std::to_string(static_cast<uint8_t>(colors.b * 255.0f)) << "\n";
	//	}
	//}
	//
	//ppmFile.flush();

	ppmFile.close();
}

Renderer::Renderer()
{
	//m_LoadedPPM = std::make_shared<ImageBuffer>();
}

void Renderer::OnResize(uint32_t width, uint32_t height)
{
	if (m_FinalImage)
	{
		if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
			return;
		m_FinalImage->Resize(width, height);
	}
	else
	{
		m_FinalImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}

	delete[] m_ImageData;
	//delete[] m_ScreenshotBuffer;
	m_ImageData = new uint32_t[width * height];
	//m_ScreenshotBuffer = new uint32_t[width * height];
}

#define THREAD 1

void async_render_func(Renderer& renderer, uint32_t width, uint32_t height, uint32_t thread_index)
{
	//if (thread_index % 2 != 0)
	//	return;
	float cx = thread_index % width;
	float cy = thread_index / width;

	float size_x = (float)width / THREAD;
	float offset_x = size_x * cx;

	float size_y = (float)height;// / 1.0f;
	float offset_y = size_y * cy;


	float ratioX = (float)renderer.m_FinalImage->GetWidth() / (float)renderer.m_LoadedPPM->width;
	float ratioY = (float)renderer.m_FinalImage->GetHeight() / (float)renderer.m_LoadedPPM->height;

	float ratio = ratioX < ratioY ? ratioX : ratioY;

	float width_s = (float)renderer.m_LoadedPPM->width * ratio;
	float height_s = (float)renderer.m_LoadedPPM->height * ratio;

	float posX = ((float)renderer.m_FinalImage->GetWidth() - ((float)renderer.m_LoadedPPM->width * ratio)) / 2.0f;
	float posY = ((float)renderer.m_FinalImage->GetHeight() - ((float)renderer.m_LoadedPPM->height * ratio)) / 2.0f;


	float cx_s = thread_index % width;
	float cy_s = thread_index / width;

	float size_x_s = (float)width / THREAD;
	float offset_x_s = size_x_s * cx_s;

	float size_y_s = (float)height;// / 1.0f;
	float offset_y_s = size_y_s * cy_s;

	//uint32_t x_offset = offset / width;
	//uint32_t y_offset = offset % width;

	//std::cout << "size " << size << " / offset " << offset << "\n";

	for (uint32_t y = static_cast<uint32_t>(offset_y); y < static_cast<uint32_t>(size_y + offset_y); y++)
	{
		//if (y >= height)
			//break;
		if (y >= height_s)
			break;
		for (uint32_t x = static_cast<uint32_t>(offset_x); x < static_cast<uint32_t>(size_x + offset_x); x++)
		{
			//if (x >= width)
				//break;

			if (x >= height_s)
				break;

			glm::vec2 coord = { x / (float)width, y / (float)height };

			//coord.x = coord.x * renderer.m_Aspect;
			//coord = coord * 2.0f - 1.0f;

			uint32_t px = x + width * y;
			uint32_t px_s = x + width_s * y;
			//renderer.m_ImageData[px] = ColorUtils::Vec4ToRGBA(renderer.PerPixel(coord));
			renderer.m_ImageData[px] = renderer.m_LoadedPPM->buffer[px_s];
		}
	}
}

void Renderer::Render()
{
	m_Aspect = m_FinalImage->GetWidth() / (float)m_FinalImage->GetHeight();

	if (!m_LoadedPPM.get() || !m_LoadedPPM->buffer)
		return;
	//uint32_t i = (coord.x * m_LoadedPPM->width) + m_LoadedPPM->width * (coord.y * m_LoadedPPM->height);
	//uint32_t a = m_LoadedPPM->buffer[i];

	std::vector<std::thread> threads;

	for (uint32_t i = 0; i < THREAD; i++) 
	{
		threads.emplace_back(async_render_func, *this, m_FinalImage->GetWidth(), m_FinalImage->GetHeight(), i);
		//threads[i].join();
	}

	for (int i = 0; i < threads.size(); i++)
	{
		//threads[i].detach();
		threads[i].join();
	}

	//threads[THREAD - 1].join();

	//std::cin.get();
	
	m_FinalImage->SetData(m_ImageData);
}

void Renderer::SaveAsPPM(const char* path)
{

	//for (uint32_t i = 0; i < m_FinalImage->GetWidth() * m_FinalImage->GetHeight(); i++)
	//{
	//	m_ScreenshotBuffer[i] = m_ImageData[i];
	//}

	//std::shared_ptr<ImageBuffer> copyImage = std::make_shared<ImageBuffer>(m_FinalImage->GetWidth(), m_FinalImage->GetHeight(), m_ScreenshotBuffer);
	//copyImage->SetData(m_ScreenshotBuffer);

	/*std::thread saveAsPPMThread(*/ load_from_ppm_func(this, path);
	//saveAsPPMThread.detach();
}

glm::vec4 Renderer::PerPixel(glm::vec2 coord)
{
	glm::vec4 color(coord, 0.0f, 1.0f);
	
	


	//float pos = posX > posY ? posX : posY;

	uint32_t i = (coord.x * m_LoadedPPM->width) + m_LoadedPPM->width * (coord.y * m_LoadedPPM->height);
	uint32_t a = m_LoadedPPM->buffer[i];

	ColorUtils::RGBAtoVec4(color, a);
	return color;
}
