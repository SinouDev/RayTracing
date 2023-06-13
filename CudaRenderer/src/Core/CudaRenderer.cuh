#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "DataBuffer2D.cuh"
#include "CudaCamera.cuh"
#include "Ray.cuh"

class CudaRenderer;
class Scene;
class SceneComponent;
class Sphere;
class Material;

namespace SGOL { class Color; }

namespace CUDA {
    /// \brief CUDA kernel function for rendering the screen
    /// \param renderer The CudaRenderer object
    __global__ void RenderCudaScreen(CudaRenderer renderer);

    /// \brief Helper function for rendering the screen using old-style threading
    /// \param renderer Reference to the CudaRenderer object
    __device__ void OldStyle_RenderCudaScreen(CudaRenderer& renderer);

    __global__ void RendererPostProcessing(CudaRenderer renderer);
}

/// \brief Class responsible for rendering a scene using CUDA
class CudaRenderer
{
private:
    using Ray = SGOL::Ray;
    using Color = SGOL::Color;

public:
    typedef SGOL::DataBuffer2D<uint32_t> ScreenDataBuffer;
    typedef SGOL::DataBuffer2D<glm::vec4> AccumulationDataBuffer;

private:
    /// \brief Struct representing the settings for the CUDA renderer
    struct Settings {
        bool accumulate = true;             ///< Flag indicating whether to accumulate samples
        bool oldStyleThreading = false;     ///< Flag indicating whether to use old-style threading
    };

public:
    /// \brief Default constructor
    CudaRenderer();

    /// \brief Constructor that initializes the renderer with the specified width and height
    /// \param width The width of the screen buffer
    /// \param height The height of the screen buffer
    CudaRenderer(size_t width, size_t height);

    /// \brief Copy constructor
    /// \param renderer The CudaRenderer object to be copied
    __host__ __device__ CudaRenderer(CudaRenderer& renderer);

    /// \brief Destructor
    ~CudaRenderer();

    /// \brief Resizes the screen buffer to the specified width and height
    /// \param width The new width of the screen buffer
    /// \param height The new height of the screen buffer
    void Resize(size_t width, size_t height);

    /// \brief Renders the scene using the specified scene and camera
    /// \param scene Pointer to the Scene object
    /// \param camera Pointer to the Camera object
    void Render(const Scene* scene, const Camera* camera);

    /// \brief Save the current buffer to a png file
    /// \param path to save the output image
    void SaveAsPNG(const char* path);

    /// \brief Save the current buffer to a ppm file (It is recommeneded to use the PNG method)
    /// \param path to save the output image
    void SaveAsPPM(const char* path);
    
    /// \brief Checks if the renderer is busy rendering
    /// \return True if the renderer is busy, false otherwise
    inline bool IsBusy() { return m_ScreenBuffer.IsBusy(); }

    /// \brief Gets the width of the screen buffer
    /// \return The width of the screen buffer
    __host__ __device__ size_t Width() { return m_ScreenBuffer.Width(); }

    /// \brief Gets the height of the screen buffer
    /// \return The height of the screen buffer
    __host__ __device__ size_t Height() { return m_ScreenBuffer.Height(); }

    /// \brief Gets a reference to the screen buffer
    /// \return Reference to the screen buffer
    __host__ __device__ ScreenDataBuffer& ScreenBuffer() { return m_ScreenBuffer; }

    /// \brief Gets the render time in seconds
    /// \return The render time in seconds
    inline float GetRenderTime() { return m_RenderTime; }

    /// \brief Gets a reference to the sampling rate
    /// \return Reference to the sampling rate
    inline uint32_t& GetSamplingRate() { return m_SamplingRate; }

    /// \brief Gets the pointer to the active Camera object (const version)
    /// \return Pointer to the active Camera object (const version)
    inline const Camera* GetActiveCamera() const { return m_ActiveCamera; }

    /// \brief Gets the pointer to the active Camera object
    /// \return Pointer to the active Camera object
    inline Camera* GetActiveCamera() { return const_cast<Camera*>(m_ActiveCamera); }

    /// \brief Gets the pointer to the active Scene object (const version)
    /// \return Pointer to the active Scene object (const version)
    inline const Scene* GetActiveScene() const { return m_ActiveScene; }

    /// \brief Gets the pointer to the active Scene object
    /// \return Pointer to the active Scene object
    inline Scene* GetActiveScene() { return const_cast<Scene*>(m_ActiveScene); }

    /// \brief Gets a reference to the accumulation threshold
    /// \return Reference to the accumulation threshold
    inline uint32_t& GetAccumulationThreshold() { return m_AccumulationThreshold; }

    inline uint32_t& GetBlurSamplingArea() { return m_BlurSamplingArea; }

    /// \brief Checks if the renderer is idle
    /// \return True if the renderer is idle, false otherwise
    inline bool IsIdle() { return m_FrameIndex > m_AccumulationThreshold || m_ActiveCamera == nullptr; }

    /// \brief Resets the frame index to 1
    __host__ __device__ void ResetFrameIndex() { m_FrameIndex = 1; }

    /// \brief Gets a reference to the renderer settings
    /// \return Reference to the renderer settings
    __host__ __device__ Settings& GetSettings() { return m_Settings; }

private:
    struct HitPayload
    {
        float hitDistance;
        glm::vec3 worldNormal;
        glm::vec3 worldPosition;
        int32_t objectIndex;
        bool front_face = false;

        /// \brief Sets the front face normal based on the given ray and outward normal
        /// \param ray The Ray object
        /// \param outward_normal The outward normal vector
        __device__ __host__ inline void set_face_normal(const Ray& ray, glm::vec3 outward_normal)
        {
            front_face = glm::dot(ray.direction, outward_normal) < 0.0f;
            worldNormal = front_face ? outward_normal : -outward_normal;
        }
    };

    struct MaterialHandleResult
    {
        bool colorScattered, lightEmitted;

        /// \brief Default constructor
        __device__ __host__ MaterialHandleResult()
            : colorScattered(false), lightEmitted(false)
        {}

        /// \brief Constructor that initializes the MaterialHandleResult object with the specified values
        /// \param colorScattered Flag indicating if color is scattered
        /// \param lightEmitted Flag indicating if light is emitted
        __device__ __host__ MaterialHandleResult(bool colorScattered, bool lightEmitted)
            : colorScattered(colorScattered), lightEmitted(lightEmitted)
        {}
    };

    // Cuda Kernel Methods:
    /// \brief Initializes the CUDA random states
    void initCudaRandomStates();

    /// \brief Performs per-pixel rendering using CUDA
    /// \param coord The coordinate of the pixel
    /// \return The color of the pixel
    __device__ Color PerPixel(glm::vec2 coord);

    /// \brief Traces a ray in the scene and returns the hit payload
    /// \param ray The Ray object to trace
    /// \return The HitPayload object containing hit information
    __device__ HitPayload TraceRay(const Ray& ray);

    /// \brief Finds the closest hit in the scene for the given ray
    /// \param ray The Ray object
    /// \param hitDistance The hit distance
    /// \param objectIndex The index of the object in the scene
    /// \return The HitPayload object containing hit information
    __device__ HitPayload ClosestHit(const Ray& ray, float hitDistance, int32_t objectIndex);

    /// \brief Handles the miss case for the given ray
    /// \param ray The Ray object
    /// \return The HitPayload object containing miss information
    __device__ HitPayload Miss(const Ray& ray);

    /// \brief Generates a ray for a specific component of the camera
    /// \param component The CameraComponent object
    /// \param coord The coordinate of the pixel
    /// \return The generated Ray object
    __device__ Ray GetRayForComponent(const CameraComponent& component, const glm::vec2& coord);

    /// \brief Handles the material for a Sphere object in the scene
    /// \param sceneComp The SceneComponent object representing the sphere
    /// \param payload The HitPayload object containing hit information
    /// \param ray The Ray object
    __device__ void HandleMaterialForSphere(const SceneComponent& sceneComp, const HitPayload& payload, Ray& ray);

    friend __global__ void CUDA::RendererPostProcessing(CudaRenderer renderer);
    friend __global__ void CUDA::RenderCudaScreen(CudaRenderer renderer);
    friend __device__ void CUDA::OldStyle_RenderCudaScreen(CudaRenderer& renderer);

private:
    ScreenDataBuffer m_ScreenBuffer;
    AccumulationDataBuffer m_AccumulationDataBuffer;
    SGOL::CuRandom m_RandomState;
    Settings m_Settings;
    uint32_t m_SamplingRate = 1;
    bool m_IsCopy = false;
    uint32_t m_FrameIndex = 1;
    uint32_t m_AccumulationThreshold = 400;
    const Camera* m_ActiveCamera = nullptr;
    const Scene* m_ActiveScene = nullptr;
    uint32_t m_BlurSamplingArea = 2;

    // not needed for shalow copy
    int32_t m_DeviceID = 0, m_SumSMs;
    float m_RenderTime = 0.0f;
};
