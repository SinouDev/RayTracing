#include "Perlin.h"

#include "Utils/Utils.h"
#include "Utils/Random.h"

using Utils::Math::Vec3;
using Utils::Math::Point3;

Perlin::Perlin()
{
	//m_RandomFloat = new float[s_PointCount];
	m_RandomVec3 = new Vec3[s_PointCount];
	for (int32_t i = 0; i < s_PointCount; ++i)
	{
		//m_RandomFloat[i] = Utils::Random::RandomFloat();
		m_RandomVec3[i] = Utils::Math::UnitVec(Utils::Random::RandomVec3(-1.0f, 1.0f));
	}

	m_PermX = PerlinGeneratePerm();
	m_PermY = PerlinGeneratePerm();
	m_PermZ = PerlinGeneratePerm();

}

Perlin::~Perlin()
{
	delete[] m_RandomVec3;
	//delete[] m_RandomFloat;
	delete[] m_PermX;
	delete[] m_PermY;
	delete[] m_PermZ;
}

float Perlin::Noise(const Point3& point) const
{
	if (!m_SmoothNoise)
	{
		int32_t i = static_cast<int32_t>(4.0f * point.x) & 0xFF;
		int32_t j = static_cast<int32_t>(4.0f * point.y) & 0xFF;
		int32_t k = static_cast<int32_t>(4.0f * point.z) & 0xFF;

		//return m_RandomFloat[m_PermX[i] ^ m_PermY[j] ^ m_PermZ[k]];
	}

	Vec3 uvw(point.x - Utils::Math::Floor(point.x), point.y - Utils::Math::Floor(point.y), point.z - Utils::Math::Floor(point.z));

	int32_t i = static_cast<int32_t>(Utils::Math::Floor(point.x));
	int32_t j = static_cast<int32_t>(Utils::Math::Floor(point.y));
	int32_t k = static_cast<int32_t>(Utils::Math::Floor(point.z));

	Vec3mat c;

	for (int32_t di = 0; di < 2; di++)
		for (int32_t dj = 0; dj < 2; dj++)
			for (int32_t dk = 0; dk < 2; dk++)
				//c[di][dj][dk] = m_RandomFloat[m_PermX[(i + di) & 0xFF] ^ m_PermY[(j + dj) & 0xFF] ^ m_PermZ[(k + dk) & 0xFF]];
				c[di][dj][dk] = m_RandomVec3[m_PermX[(i + di) & 0xFF] ^ m_PermY[(j + dj) & 0xFF] ^ m_PermZ[(k + dk) & 0xFF]];

	return TrilinearInterp(c, uvw);

}

float Perlin::Turbulence(const Point3& point, int32_t depth) const
{
	float accum = 0.0f;
	Point3 tmp_point = point;
	float weight = 1.0f;

	for (int32_t i = 0; i < depth; i++)
	{
		accum += weight * Noise(tmp_point);
		weight *= 0.5f;
		tmp_point *= 2.0f;
	}

	return Utils::Math::Abs(accum);
}

//float Perlin::TrilinearInterp(const Cmat& c, const Vec3& uvw)
//{
//	float accum = 0.0f;
//	for (int32_t i = 0; i < 2; i++)
//		for (int32_t j = 0; j < 2; j++)
//			for (int32_t k = 0; k < 2; k++)
//				accum += (i * uvw.s + (1 - i) * (1 - uvw.s)) * (j * uvw.t + (1 - j) * (1 - uvw.t)) * (k * uvw.p + (1 - k) * (1 - uvw.p)) * c[i][j][k];
//
//	return accum;
//}

float Perlin::TrilinearInterp(const Vec3mat& c, const Vec3& uvw)
{

	Vec3 tmp;
	tmp.s = uvw.s * uvw.s * (3.0f - 2.0f * uvw.s);
	tmp.t = uvw.t * uvw.t * (3.0f - 2.0f * uvw.t);
	tmp.p = uvw.p * uvw.p * (3.0f - 2.0f * uvw.p);

	float accum = 0.0f;
	for (int32_t i = 0; i < 2; i++)
		for (int32_t j = 0; j < 2; j++)
			for (int32_t k = 0; k < 2; k++)
			{
				Vec3 weightV(tmp.s - i, tmp.t - j, tmp.p - k);
				accum += (i * tmp.s + (1 - i) * (1 - tmp.s)) * (j * tmp.t + (1 - j) * (1 - tmp.t)) * (k * tmp.p + (1 - k) * (1 - tmp.p)) * Utils::Math::Dot(c[i][j][k], weightV);
			}

	return accum;
}

int32_t* Perlin::PerlinGeneratePerm()
{
	int32_t* perm = new int32_t[s_PointCount];

	for (int32_t i = 0; i < s_PointCount; i++)
	{
		perm[i] = i;
	}

	Permute(perm, s_PointCount);

	return perm;
}

void Perlin::Permute(int32_t* perm, int32_t n)
{
	for (int32_t i = n - 1; i > 0; i--)
	{
		int32_t target = Utils::Random::RandomInt(0, i);
		int32_t tmp = perm[i];
		perm[i] = perm[target];
		perm[target] = tmp;
	}
}
