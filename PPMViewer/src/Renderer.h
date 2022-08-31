#pragma once

#include "glm/glm.hpp"

#include "Walnut/Image.h"

#include <memory>



class Renderer
{
public:

	Renderer();

	void OnResize(uint32_t width, uint32_t height);
	void Render();

	void SaveAsPPM(const char* path);

	std::shared_ptr<Walnut::Image> GetFinalImage() { return m_FinalImage; }

	friend void async_render_func(Renderer&, uint32_t, uint32_t, uint32_t);

private:

	glm::vec4 PerPixel(glm::vec2 coord);

private:

	struct ImageBuffer;

	friend void load_from_ppm_func(Renderer*, const char*);

private:

	float m_Aspect = 0.0f;

	uint32_t* m_ImageData = nullptr;
	std::shared_ptr<ImageBuffer> m_LoadedPPM;
	std::shared_ptr<Walnut::Image> m_FinalImage;
};

