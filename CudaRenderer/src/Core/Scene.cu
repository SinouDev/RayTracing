#include "Scene.cuh"

void Scene::AddSphere(const Sphere& sphere)
{
	m_Spheres.push_back(sphere);
	
	m_SceneComponent.spheres = m_Spheres.begin().base().get();
	m_SceneComponent.spheres_size = m_Spheres.size();
}

void Scene::AddMaterial(const Material& material)
{
	m_Materials.push_back(material);

	m_SceneComponent.materials = m_Materials.begin().base().get();
	m_SceneComponent.material_size = m_Materials.size();
}

__device__ __host__ thrust::universal_vector<Sphere>& Scene::Spheres()
{
	return m_Spheres;
}

__device__ __host__ thrust::universal_vector<Material>& Scene::Materials()
{
	return m_Materials;
}