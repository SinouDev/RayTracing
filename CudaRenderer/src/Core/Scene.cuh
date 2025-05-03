#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>

#include "gmath.cuh"
#include "Ray.cuh"
#include "CudaColor.cuh"

/// \brief Struct representing a sphere in the scene
struct Sphere {
    glm::vec3 position;     ///< Position of the sphere
    float radius;           ///< Radius of the sphere
    int32_t materialIndex;  ///< Index of the material assigned to the sphere
    bool draw;              ///< Flag indicating whether to draw the sphere
};

/// \brief Enum representing the type of material
enum MaterialType : uint32_t {
    Material_Lambertian = 0,    ///< Lambertian material type
    Material_Metal,             ///< Metal material type
    Material_Dilectric,         ///< Dilectric material type
    Material_Emissve,           ///< Emissive material type
    Material_Isotropic,         ///< Isotropic material type
};

/// \brief Struct representing a material in the scene
struct Material {
    glm::vec3 albedo;       ///< Albedo of the material
    glm::vec3 emition;      ///< Emission of the material

    union {
        float roughness;        ///< Roughness property (for Metal material)
        float melatic;          ///< Melatic property (for Dilectric material)
        float refractionIndex;  ///< Refraction index (for Dilectric material)
        float lightIntencity;   ///< Light intensity (for Emissive material)
        float property;         ///< Generic material property
    };

    MaterialType type;      ///< Type of the material
};

/// \brief Struct representing the components of the scene
struct SceneComponent {
    const Sphere* spheres = nullptr;                ///< Array of spheres in the scene
    const Material* materials = nullptr;            ///< Array of materials in the scene
    size_t spheres_size = 0;                         ///< Size of the spheres array
    size_t material_size = 0;                        ///< Size of the materials array
    glm::vec3 lightDirection { -1.0f };              ///< Light direction vector
};

/// \brief Class representing the scene
class Scene {
public:
    /// \brief Adds a sphere to the scene
    /// \param sphere The sphere to add
    void AddSphere(const Sphere& sphere);

    /// \brief Adds a material to the scene
    /// \param material The material to add
    void AddMaterial(const Material& material);

    /// \brief Gets a reference to the vector of spheres in the scene
    /// \return Reference to the vector of spheres
    __device__ __host__ thrust::universal_vector<Sphere>& Spheres();

    /// \brief Gets a reference to the vector of materials in the scene
    /// \return Reference to the vector of materials
    __device__ __host__ thrust::universal_vector<Material>& Materials();

    /// \brief Gets a const reference to the scene component
    /// \return Const reference to the scene component
    __device__ __host__ const SceneComponent& GetSceneComponent() const { return m_SceneComponent; }

    /// \brief Gets a reference to the scene component
    /// \return Reference to the scene component
    __device__ __host__ SceneComponent& GetSceneComponent() { return m_SceneComponent; }

    /// \brief Gets a const reference to the light direction vector
    /// \return Const reference to the light direction vector
    __device__ __host__ const glm::vec3& GetLightDirection() const { return m_SceneComponent.lightDirection; }

    /// \brief Gets a reference to the light direction vector
    /// \return Reference to the light direction vector
    __device__ __host__ glm::vec3& GetLightDirection() { return m_SceneComponent.lightDirection; }

    /// \brief Gets a reference to the ray bouncing rate
    /// \return Reference to the ray bouncing rate
    __device__ __host__ uint32_t& GetRayBouncingRate() { return m_RayBouncingRate; }

    /// \brief Gets the ray bouncing rate
    /// \return The ray bouncing rate
    __device__ __host__ uint32_t GetRayBouncingRate() const { return m_RayBouncingRate; }

    /// \brief Gets a reference to the starting ambient light color for rays
    /// \return Reference to the starting ambient light color
    __device__ __host__ SGOL::Color& GetRayAmbientLightColorStart() { return m_AmbientLightColorStart; }

    /// \brief Gets a reference to the ending ambient light color for rays
    /// \return Reference to the ending ambient light color
    __device__ __host__ SGOL::Color& GetRayAmbientLightColorEnd() { return m_AmbientLightColorEnd; }

    /// \brief Overloaded new operator for CUDA memory allocation
    /// \param size Size of the memory to allocate
    /// \return Pointer to the allocated memory
    void* operator new(size_t size)
    {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    /// \brief Overloaded delete operator for CUDA memory deallocation
    /// \param ptr Pointer to the memory to deallocate
    void operator delete(void* ptr)
    {
        cudaFree(ptr);
    }

    /// \brief Gets the sky color for a given ray
    /// \param ray The ray
    /// \return The sky color
    __device__ __host__ SGOL::Color GetSkyColor(const SGOL::Ray& ray) const
    {
        glm::vec3 unit_direction = SGOL::gmath::unit_vec(ray.direction);
        float t = 0.5f * (unit_direction.y + 1.0f);
        return (1.0f - t) * m_AmbientLightColorEnd + t * m_AmbientLightColorStart;
    }

    /// \brief Gets the name of a material type
    /// \param material The material
    /// \return The name of the material type
    static __host__ const char* MaterailTypeName(Material& material)
    {
        switch (material.type)
        {
        case Material_Metal:
            return "Metal";
        case Material_Lambertian:
            return "Lambertian";
        case Material_Dilectric:
            return "Dilectric";
        case Material_Emissve:
            return "Emissive";
        case Material_Isotropic:
            return "Isotropic";
        default:
            return "";
        }
    }

    /// \brief Creates a material with the specified properties
    /// \param type The type of the material
    /// \param albedo The albedo of the material
    /// \param emition The emission of the material
    /// \param materialProperty The material property
    /// \return The created material
    static inline __host__ __device__ Material CreateMaterial(MaterialType type, const glm::vec3& albedo = glm::vec3(0.0f), const glm::vec3& emition = glm::vec3(0.0f), float materialProperty = 0.0f)
    {
        Material material;
        material.type = type;
        material.albedo = albedo;
        material.emition = emition;
        material.property = materialProperty;
        return material;
    }

private:
    SceneComponent m_SceneComponent;                            ///< Scene component
    thrust::universal_vector<Sphere> m_Spheres;                 ///< Vector of spheres in the scene
    thrust::universal_vector<Material> m_Materials;             ///< Vector of materials in the scene
    uint32_t m_RayBouncingRate = 5;                             ///< Ray bouncing rate
    SGOL::Color m_AmbientLightColorStart {0.517f, 0.502f, 1.0f}; ///< Starting ambient light color for rays
    SGOL::Color m_AmbientLightColorEnd {0.98f, 0.878f, 0.584f};  ///< Ending ambient light color for rays
};
