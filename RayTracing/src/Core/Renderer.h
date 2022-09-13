#pragma once

#include "glm/glm.hpp"

#include "Object/HittableObjectList.h"

#include <memory>
#include <atomic>
#include <functional>

class Camera;
class Ray;

class Renderer
{

private:

	struct ThreadScheduler {
		bool completed = false;
		bool rendering = false;
		float offset_x = 0, offset_y = 0;
		uint32_t n_width = 0, n_height = 0;
		//uint32_t done_x = 0, done_y = 0;

		void Set(bool c, bool r, float off_x, float off_y, uint32_t n_w, uint32_t n_h/*, uint32_t d_x, uint32_t d_y*/)
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

	struct ImageBuffer {
		uint32_t width = 0, height = 0;
		uint8_t channels = 0;
		std::atomic<uint8_t*> buffer = nullptr;

		ImageBuffer() = default;

		ImageBuffer(uint32_t w, uint32_t h, uint8_t c, uint8_t* b)
			: width(w), height(h), channels(c), buffer(b)
		{}

		~ImageBuffer()
		{
			delete[] buffer.load();
		}

		void Resize(uint32_t w, uint32_t h, uint8_t c)
		{
			delete[] buffer.load();
			buffer.store(nullptr);
			width = w;
			height = h;
			channels = c;
			buffer.store(new uint8_t[width * height * channels]);
		}

		void Clear()
		{
			for (uint32_t i = 0; i < width * height * channels; i++)
				buffer.load()[i] = 0x0;
		}

		uint8_t operator[](int i) const
		{
			return buffer.load()[i];
		}

		uint8_t& operator[](int i)
		{
			return buffer.load()[i];
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
			return buffer.load();
		}

		template<typename T>
		T Get()
		{
			return (T)buffer.load();
		}

	};

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

	void SetScalingEnabled(bool enable);

	void SetWorkingThreads(uint32_t threads);

	inline uint32_t GetThreadCount() { return m_ThreadCount; }
	inline uint32_t& GetSamplingRate() { return m_SamplingRate; }
	inline int32_t& GetRayColorDepth() { return m_RayColorDepth; }

	inline HittableObjectList& GetHittableObjectList() { return m_HittableObjectList; }

	static glm::vec3& GetRayBackgroundColor();
	static glm::vec3& GetRayBackgroundColor1();

	inline const ImageBufferPtr& GetImageDataBuffer() const { return m_ImageData; }
	inline bool IsRendering() { return m_AsyncThreadRunning; }

	inline float GetRenderingTime() { return m_RenderingTime; }

	inline std::atomic_bool& IsClearingOnEachFrame() { return m_ClearOnEachFrame; }
	inline uint64_t& GetClearDelay() { return m_ClearDelay; }

	void SetClearOnEachFrame(bool clear);

	void SetClearDelay(uint64_t ms);

	void ClearScene();


private:

	friend void save_as_ppm_func(const char*, std::shared_ptr<Renderer::ImageBuffer>&);
	friend void async_render_func(Renderer&, const std::shared_ptr<Camera>&, uint32_t, uint32_t, uint32_t);

	void Render(const std::shared_ptr<Camera>& camera);
	void ResizeThreadScheduler();

	glm::vec4 RayTrace(Ray& ray);

private:

	//
	std::shared_ptr<std::vector<ThreadScheduler>> m_ThreadScheduler;

	//
	RenderingCompleteCallback m_ThreadDoneCallBack;

	//
	uint32_t m_ThreadCount = 8;

	//
	float m_Aspect = 0.0f;

	//
	uint32_t m_SamplingRate = 1;

	//
	int32_t m_RayColorDepth = 10;

	//
	uint8_t m_ScreenshotChannels = 4;

	//
	float m_RenderingTime = 0.0f;

	// main image buffer that will show in the screen
	ImageBufferPtr m_ImageData;

	// a
	ImageBufferPtr m_ScreenshotBuffer;

	// The hittiable object list that will be ray traced
	HittableObjectList m_HittableObjectList;

	// used to check if the renderer is ready
	bool m_RendererReady = false; 

	// not used yet
	bool m_ScalingEnabled = false;

	// used to indicate if the renderer is currently rendering
	std::atomic_bool m_AsyncThreadRunning = false; 

	// this flag is set to false when to make all threads stop rendering
	std::atomic_bool m_AsyncThreadFlagRunning = false;

	// used to recylce the same threads without createing new one NOTE: it's still not fully tested
	std::atomic_bool m_AsyncThreadRecycleFlag = false;

	// this flag is used to determine if it is rendered once or no
	std::atomic_bool m_AsyncThreadRenderOneFlag = false;

	// used to clear the frame after a delay, same as glClearColor method for opengl
	std::atomic_bool m_ClearOnEachFrame = false; 

	// used for the clear delay in ms 
	uint64_t m_ClearDelay = 500U;

};