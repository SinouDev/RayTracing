#pragma once

#include "glm/glm.hpp"

#include "Object/HittableObjectList.h"

#include "Utils/Math.h"

#include <memory>
#include <atomic>
#include <functional>
#include <thread>

class Camera;
class Ray;

class Renderer
{

private:

	/// <summary>
	/// Thread scheduler structure to help in multithreading rendering
	/// </summary>
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

	/// <summary>
	/// This structure holds the buffer data that the renderer need
	/// </summary>
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

	/// <summary>
	/// 
	/// </summary>
	Renderer();

	~Renderer();

	/// <summary>
	/// 
	/// </summary>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="resizeDoneCallback"></param>
	void OnResize(uint32_t width, uint32_t height, RenderingCompleteCallback resizeDoneCallback = nullptr);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="camera"></param>
	void RenderOnce(const std::shared_ptr<Camera>& camera);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="camera"></param>
	void StartAsyncRender(const std::shared_ptr<Camera>& camera);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="callBack"></param>
	void StopRendering(RenderingCompleteCallback callBack = nullptr);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="path"></param>
	void SaveAsPPM(const char* path);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="path"></param>
	void SaveAsPNG(const char* path);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="enable"></param>
	void SetScalingEnabled(bool enable);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="threads"></param>
	void SetWorkingThreads(uint32_t threads);

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline uint32_t GetThreadCount() { return m_ThreadCount; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline uint32_t& GetSamplingRate() { return m_SamplingRate; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline int32_t& GetRayColorDepth() { return m_RayColorDepth; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline HittableObjectList& GetHittableObjectList() { return m_HittableObjectList; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	static Utils::Math::Color3& GetRayBackgroundColor();

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	static Utils::Math::Color3& GetRayBackgroundColor1();

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline const ImageBufferPtr& GetImageDataBuffer() const { return m_ImageData; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline bool IsRendering() { return m_AsyncThreadRunning; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline float GetRenderingTime() { return m_RenderingTime; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline std::atomic_bool& IsClearingOnEachFrame() { return m_ClearOnEachFrame; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	inline uint64_t& GetClearDelay() { return m_ClearDelay; }

	/// <summary>
	/// 
	/// </summary>
	/// <param name="clear"></param>
	void SetClearOnEachFrame(bool clear);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ms"></param>
	void SetClearDelay(uint64_t ms);

	/// <summary>
	/// 
	/// </summary>
	void ClearScene();

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	static const inline uint32_t GetMaximumThreads()
	{
		static uint32_t nthreads = std::thread::hardware_concurrency();// copied from https://stackoverflow.com/a/150971/10782228
		if (nthreads == 0)
			nthreads = 8; // as most consumer PC have 4 cores and 8 threads
		return nthreads;
	}


private:
	
	/// <summary>
	/// 
	/// </summary>
	/// <param name="path"></param>
	/// <param name="image"></param>
	friend void save_as_ppm_func(const char* path, std::shared_ptr<Renderer::ImageBuffer>& image);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="renderer"></param>
	/// <param name="camera"></param>
	/// <param name="width"></param>
	/// <param name="height"></param>
	/// <param name="thread_index"></param>
	friend void async_render_func(Renderer& renderer, const std::shared_ptr<Camera>& camera, uint32_t width, uint32_t height, uint32_t thread_index);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="camera"></param>
	void Render(const std::shared_ptr<Camera>& camera);

	/// <summary>
	/// 
	/// </summary>
	void ResizeThreadScheduler();

	/// <summary>
	/// 
	/// </summary>
	/// <param name="ray"></param>
	/// <returns></returns>
	Utils::Math::Color4 RayTrace(Ray& ray);

private:

	// renderer thread scheduler for switching threads between screen parts
	std::shared_ptr<std::vector<ThreadScheduler>> m_ThreadScheduler;

	// called when all threads is done renderering the current frame
	RenderingCompleteCallback m_ThreadDoneCallBack;

	// count of working threads work by half of threads count aka cores count by default 
	uint32_t m_ThreadCount = GetMaximumThreads() / 2;

	// scheduler multiplier
	uint32_t m_SchedulerMultiplier = 11;

	// rendered image aspect ratio
	float m_Aspect = 1.0f;

	// ray tracer sampling rate per pixel
	uint32_t m_SamplingRate = 1;

	// ray tracer color depth (light bouncing)
	int32_t m_RayColorDepth = 10;

	// screenshot buffer's channels
	uint8_t m_ScreenshotChannels = 4;

	// the time it tooks of the rendering
	float m_RenderingTime = 0.0f;

	// main image buffer that will show in the screen
	ImageBufferPtr m_ImageData;

	// screenshot buffer for saving it later
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