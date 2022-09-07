#pragma once

#include "glm/glm.hpp"

class Perlin
{
	using Color = glm::vec3;
	using Vec3 = glm::vec3;
	using Point3 = glm::vec3;
	using Cmat = float[2][2][2];
	using Vec3mat = Vec3[2][2][2];

public:

	Perlin();
	~Perlin();

	float Noise(const Point3& point) const;
	float Turbulence(const Point3& point, int32_t depth = 7) const;

private:
	
	//static float TrilinearInterp(const Cmat& c, const Vec3& uvw);
	static float TrilinearInterp(const Vec3mat& c, const Vec3& uvw);
	static int32_t* PerlinGeneratePerm();
	static void Permute(int32_t* perm, int32_t n);

private:

	const static int32_t s_PointCount = 0xFF + 1;

	Vec3* m_RandomVec3 = nullptr;
	//float* m_RandomFloat = nullptr;
	int32_t* m_PermX = nullptr;
	int32_t* m_PermY = nullptr;
	int32_t* m_PermZ = nullptr;

	bool m_SmoothNoise = true;
	bool m_HermitianSmoothing = true;
};