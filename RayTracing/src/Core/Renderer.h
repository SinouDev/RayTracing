#pragma once

#include "glm/glm.hpp"

#include "Walnut/Image.h"

#include "Object/HittableObjectList.h"
#include "Object/Sphere.h"

#include "Material/Lambertian.h"
#include "Material/Metal.h"
#include "Material/Dielectric.h"

#include <memory>
#include <atomic>
#include <functional>

class Camera;
class Ray;

class Renderer
{

private:

	typedef struct ThreadScheduler {
		bool completed = false;
		bool rendering = false;
		float offset_x = 0, offset_y = 0;
		uint32_t n_width = 0, n_height = 0;
		uint32_t done_x = 0, done_y = 0;

		void Set(bool c, bool r, float off_x, float off_y, uint32_t n_w, uint32_t n_h, uint32_t d_x, uint32_t d_y)
		{
			completed = c;
			rendering = r;
			offset_x = off_x;
			offset_y = off_y;
			n_width = n_w;
			n_height = n_h;
			//done_x = d_x;
			//done_y = d_y;
		}

	};

	typedef struct ImageBuffer {
		uint32_t width, height;
		uint8_t channels;
		std::atomic<uint8_t*> buffer = nullptr;

		ImageBuffer() = default;

		ImageBuffer(uint32_t w, uint32_t h, uint8_t c, uint8_t* b)
			: width(w), height(h), channels(c), buffer(b)
		{}

		~ImageBuffer()
		{
			delete[] buffer;
		}

		void Resize(uint32_t w, uint32_t h, uint8_t c)
		{
			delete[] buffer;
			buffer = nullptr;
			width = w;
			height = h;
			channels = c;
			buffer = new uint8_t[width * height * channels];
		}

		void Clear()
		{
			for (uint32_t i = 0; i < width * height * channels; i++)
				buffer[i] = 0x0;
		}

		uint8_t operator[](int i) const
		{
			return buffer[i];
		}

		uint8_t& operator[](int i)
		{
			return buffer[i];
		}

		operator uint64_t* () const
		{
			return (uint64_t*)buffer.load();
		}

		operator uint32_t* () const
		{
			return (uint32_t*)buffer.load();
		}

		operator uint16_t* () const
		{
			return (uint16_t*)buffer.load();
		}

		operator uint8_t* () const
		{
			return buffer;
		}

		template<typename T>
		T& Get()
		{
			return (T&)buffer;
		}

	};

	using MaterialPtr = std::shared_ptr<Material>;
	using SpherePtr = std::shared_ptr<Sphere>;
	using ImageBufferPtr = std::shared_ptr<ImageBuffer>;

public:

	using RenderingCompleteCallback = std::function<void(void)>;

	Renderer();
	~Renderer();

	void OnResize(uint32_t width, uint32_t height, RenderingCompleteCallback resizeDoneCallback = nullptr);
	void RenderOnce(const std::shared_ptr<Camera>& camera);
	void StartAsyncRender(const std::shared_ptr<Camera>& camera);
	void StopRendering(RenderingCompleteCallback callBack = nullptr);

	void SaveAsPPM(const char* path);
	void SaveAsPNG(const char* path);

	//std::shared_ptr<Walnut::Image> GetFinalImage() { return m_FinalImage; }

	void SetScalingEnabled(bool enable);

	void SetWorkingThreads(uint32_t threads);

	inline uint32_t GetThreadCount() { return m_ThreadCount; }
	inline uint32_t& GetSamplingRate() { return m_SamplingRate; }
	inline int32_t& GetRayColorDepth() { return m_RayColorDepth; }

	inline Lambertian* get_back_shpere() { return dynamic_cast<Lambertian*>(back_shpere.get()); }
	inline Lambertian* get_center_sphere() { return dynamic_cast<Lambertian*>(center_sphere.get()); }
	inline Metal* get_left_sphere() { return dynamic_cast<Metal*>(left_sphere.get()); }
	inline Metal* get_right_sphere() { return dynamic_cast<Metal*>(right_sphere.get()); }
	inline ShinyMetal* get_small_sphere() { return dynamic_cast<ShinyMetal*>(small_sphere.get()); }
	inline Dielectric* get_glass_sphere() { return dynamic_cast<Dielectric*>(glass_sphere.get()); }

	inline SpherePtr& GetGlassSphere() { return m_GlassSphere; }
	inline glm::vec3& GetLightDir() { return m_LightDir; }

	static glm::vec3& GetRayBackgroundColor();
	static glm::vec3& GetRayBackgroundColor1();

	inline const ImageBufferPtr& GetImageDataBuffer() const { return m_ImageData; }
	inline bool IsRendering() { return m_AsyncThreadRunning; }

	inline std::atomic_bool& IsClearingOnEachFrame() { return m_ClearOnEachFrame; }
	inline uint64_t& GetClearDelay() { return m_ClearDelay; }

	void SetClearOnEachFrame(bool clear);

	void SetClearDelay(uint64_t ms);

	void ClearScene();


private:

	friend void save_as_ppm_func(const char*, std::shared_ptr<Renderer::ImageBuffer>&);
	friend void async_render_func(Renderer&, Camera&, uint32_t, uint32_t, uint32_t);
	friend void scenes(Renderer&, int32_t);

	void Render(Camera& camera);
	void Render(const std::shared_ptr<Camera>& camera);
	void ResizeThreadScheduler();

	glm::vec4 RayTrace(Ray& ray);

private:

	std::shared_ptr<std::vector<ThreadScheduler>> m_ThreadScheduler;

	RenderingCompleteCallback m_ThreadDoneCallBack;

	glm::vec3 m_LightDir = glm::vec3(1.0f, 10.0f, 3.0f);

	uint32_t m_ThreadCount = 8;

	float m_Aspect = 0.0f;
	uint32_t m_SamplingRate = 1;
	int32_t m_RayColorDepth = 10;

	uint8_t m_ScreenshotChannels = 4;

	ImageBufferPtr m_ImageData;
	ImageBufferPtr m_PreviewImageBuffer;
	ImageBufferPtr m_ScreenshotBuffer;
	//std::shared_ptr<Walnut::Image> m_FinalImage;

	HittableObjectList m_HittableObjectList;

	bool m_RendererReady = false;
	bool m_ScalingEnabled = false;
	std::atomic_bool m_AsyncThreadRunning = false;
	std::atomic_bool m_AsyncThreadFlagRunning = false;
	std::atomic_bool m_ClearOnEachFrame = false;

	uint64_t m_ClearDelay = 500U; // ms

	SpherePtr m_GlassSphere;


	MaterialPtr back_shpere;// = std::make_shared<Lambertian>(glm::vec3(0.8f, 0.8f, 0.0f));
	MaterialPtr center_sphere;// = std::make_shared<Lambertian>(glm::vec3(0.7f, 0.3f, 0.3f));
	MaterialPtr left_sphere;// = std::make_shared<Metal>(glm::vec3(0.8f, 0.8f, 0.8f), 0.3f);
	MaterialPtr right_sphere;// = std::make_shared<Metal>(glm::vec3(0.8f, 0.6f, 0.2f), 1.0f);
	MaterialPtr small_sphere;// = std::make_shared<ShinyMetal>(glm::vec3(1.0f, 0.6f, 0.0f));
	MaterialPtr glass_sphere;
};

