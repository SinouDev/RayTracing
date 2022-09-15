#pragma once

#include "glm/glm.hpp"

#include "Utils/Math.h"

class Perlin
{
	using Cmat = float[2][2][2];
	using Vec3mat = Utils::Math::Vec3[2][2][2];

public:

	Perlin();
	~Perlin();

	float Noise(const Utils::Math::Point3& point) const;
	float Turbulence(const Utils::Math::Point3& point, int32_t depth = 7) const;

private:
	
	//static float TrilinearInterp(const Cmat& c, const Vec3& uvw);
	static float TrilinearInterp(const Vec3mat& c, const Utils::Math::Vec3& uvw);
	static int32_t* PerlinGeneratePerm();
	static void Permute(int32_t* perm, int32_t n);

private:

	const static int32_t s_PointCount = 0xFF + 1;

	Utils::Math::Vec3* m_RandomVec3 = nullptr;
	//float* m_RandomFloat = nullptr;
	int32_t* m_PermX = nullptr;
	int32_t* m_PermY = nullptr;
	int32_t* m_PermZ = nullptr;

	bool m_SmoothNoise = true;
	bool m_HermitianSmoothing = true;
};