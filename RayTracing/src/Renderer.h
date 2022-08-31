#pragma once

#include "glm/glm.hpp"

#include "Walnut/Image.h"

#include "Ray.h"
#include "Camera.h"
#include "Object/HittableObjectList.h"
#include "Object/Sphere.h"

#include "Material/Lambertian.h"
#include "Material/Metal.h"
#include "Material/Dielectric.h"

#include <memory>

class Renderer
{

private:

	using MaterialPtr = std::shared_ptr<Material>;
	using SpherePtr = std::shared_ptr<Sphere>;

	typedef struct ImageBuffer {
		uint32_t width, height;
		uint32_t* buffer;

		ImageBuffer() = default;

		ImageBuffer(uint32_t w, uint32_t h, uint32_t* b)
			: width(w), height(h), buffer(b)
		{}

	};

public:

	Renderer();

	void OnResize(uint32_t width, uint32_t height, uint32_t scale_width = 0, uint32_t scale_height = 0);
	void Render(Camera& camera);

	void SaveAsPPM(const char* path);
	void SaveAsPNG(const char* path);

	std::shared_ptr<Walnut::Image> GetFinalImage() { return m_FinalImage; }

	void SetScalingEnabled(bool enable);

	inline uint32_t* GetThreadCount() { return &m_ThreadCount; }
	inline uint32_t* GetSamplingRate() { return &m_SamplingRate; }
	inline int32_t* GetRayColorDepth() { return &m_RayColorDepth; }

	inline Lambertian* get_back_shpere() { return dynamic_cast<Lambertian*>(back_shpere.get()); }
	inline Lambertian* get_center_sphere() { return dynamic_cast<Lambertian*>(center_sphere.get()); }
	inline Metal* get_left_sphere() { return dynamic_cast<Metal*>(left_sphere.get()); }
	inline Metal* get_right_sphere() { return dynamic_cast<Metal*>(right_sphere.get()); }
	inline ShinyMetal* get_small_sphere() { return dynamic_cast<ShinyMetal*>(small_sphere.get()); }
	inline Dielectric* get_glass_sphere() { return dynamic_cast<Dielectric*>(glass_sphere.get()); }
	inline SpherePtr& GetGlassSphere() { return m_GlassSphere; }
	inline glm::vec3& GetRayBackgroundColor() { return m_RayBackgroundColor; }
	inline glm::vec3& GetRayBackgroundColor1() { return m_RayBackgroundColor1; }
	inline glm::vec3& GetLightDir() { return m_LightDir; }

private:

	friend void save_as_ppm_func(const char*, std::shared_ptr<Renderer::ImageBuffer>&);
	friend void async_render_func(Renderer&, Camera&, uint32_t, uint32_t, uint32_t);
	friend void p(Renderer&);

	glm::vec4 RayTrace(Ray& ray);

private:

	glm::vec3 m_RayBackgroundColor = glm::vec3(0.5f, 0.7f, 1.0f);
	glm::vec3 m_RayBackgroundColor1 = glm::vec3(1.0, 1.0, 1.0f);
	glm::vec3 m_LightDir = glm::vec3(1.0f, 10.0f, 3.0f);

	uint32_t m_ThreadCount = 6;

	float m_Aspect = 0.0f;
	uint32_t m_SamplingRate = 1;
	int32_t m_RayColorDepth = 10;

	uint8_t m_ScreenshotChannels = 4;

	uint32_t* m_ImageData = nullptr;
	ImageBuffer m_PreviewImageBuffer;
	uint8_t* m_ScreenshotBuffer = nullptr;
	std::shared_ptr<Walnut::Image> m_FinalImage;

	HittableObjectList m_HittableObjectList;

	bool m_RendererReady = false;
	bool m_ScalingEnabled = false;

	SpherePtr m_GlassSphere;


	MaterialPtr back_shpere;// = std::make_shared<Lambertian>(glm::vec3(0.8f, 0.8f, 0.0f));
	MaterialPtr center_sphere;// = std::make_shared<Lambertian>(glm::vec3(0.7f, 0.3f, 0.3f));
	MaterialPtr left_sphere;// = std::make_shared<Metal>(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f);
	MaterialPtr right_sphere;// = std::make_shared<Metal>(glm::vec3(0.8f, 0.6f, 0.2f), 1.0f);
	MaterialPtr small_sphere;// = std::make_shared<ShinyMetal>(glm::vec3(1.0f, 0.6f, 0.0f));
	MaterialPtr glass_sphere;
};

