#pragma once

#include "SGOL/Array2D.hpp"

#include "cuda_runtime.h"
#include <stdint.h>

#include "glm/glm.hpp"
#include "glm/gtx/fast_square_root.hpp"
#include "glm/gtc/constants.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace SGOL {

	namespace gmath {

		static __host__ __device__ float fastInverseSqrt(float n)
		{
			long i;
			float x2, y;

			x2 = n * 0.5f;
			y = n;
			i = *(long*)&y;
			i = 0x5f3759df - (i >> 1); // wtf
			y = *(float*)&i;
			y = y * (1.5f - (x2 * y * y));
			y = y * (1.5f - (x2 * y * y));
			y = y * (1.5f - (x2 * y * y));
			//y = y * (1.5f - (x2 * y * y));
			//y = y * (1.5f - (x2 * y * y));
			//y = y * (1.5f - (x2 * y * y));

			return y;
		}

		static __host__ __device__ float fastSqrt(float n)
		{
			return 1.0f / fastInverseSqrt(n);
		}

#pragma pack(push)
#pragma pack(1)

		template<typename T, uint8_t size>
		struct vec {};

#if 0
		template<typename T>
		struct vec<T, 2>
		{
		protected:
			union
			{
				T t[size] = {};
			};
		public:
			__device__ __SGOL_INLINE T& __SGOL_FASTCALL operator[](size_t i)
			{
				return t[i];
			}
		};
#endif

		template<>
		struct vec<float, 2>
		{
		public:

			union {
				struct {
					float m_Dim[2];
				};

				struct {
					float x, y;
				};

				struct {
					float r, g;
				};
			};
#if 0
			union
			{
				float x, r;
			};
			union
			{
				float y, g;
			};
#endif

		public:

			__device__ vec()
				: vec(0.0f)
			{}

			__device__ vec(float x)
				: vec(x, x)
			{}

			__device__ vec(float x, float y)
				: x(x), y(y)
			{}

			__device__ vec(const vec& a)
				: vec(a.x, a.y)
			{}

			template<typename T>
			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator* (const T& a) const
			{
				vec tmp;
				tmp.x = x * a;
				tmp.y = y * a;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator* (const vec& a) const
			{
				vec tmp;
				tmp.x = x * a.x;
				tmp.y = y * a.y;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator/ (const vec& a) const
			{
				vec tmp;
				tmp.x = x / a.x;
				tmp.y = y / a.y;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator+ (const vec& a) const
			{
				vec tmp;
				tmp.x = x + a.x;
				tmp.y = y + a.y;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator- (const vec& a) const
			{
				vec tmp;
				tmp.x = x - a.x;
				tmp.y = y - a.y;
				return tmp;
			}

			__device__ __SGOL_INLINE float& __SGOL_FASTCALL operator[](size_t i)
			{
				return m_Dim[i];
			}

		};

		template<>
		struct vec<float, 3>
		{
		public:
			//float m_Dim[3];
			union {
				struct {
					float m_Dim[3];
				};

				struct {
					float x, y, z;
				};

				struct {
					float r, g, b;
				};
			};

#if 0
			union
			{
				float x, r;
			};
			union
			{
				float y, g;
			};

			union
			{
				float z, b;
			};
#endif

		public:

			__device__ vec()
				: vec(0.0f)
			{}

			__device__ vec(float x)
				: vec(x, x, x)
			{}
			__device__ vec(float x, float y, float z)
				: x(x), y(y), z(z)
			{}

			__device__ vec(const vec& a)
				: vec(a.x, a.y, a.z)
			{}

			__device__ vec(vec& a)
				: vec(a.x, a.y, a.z)
			{}

			template<typename T>
			__device__ __SGOL_INLINE vec& __SGOL_FASTCALL operator*= (const T& a)
			{
				this->x *= a;
				this->y *= a;
				this->z *= a;
				return *this;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator* (const vec& a) const
			{
				vec tmp;
				tmp.x = x * a.x;
				tmp.y = y * a.y;
				tmp.z = z * a.z;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator/ (const vec& a) const
			{
				vec tmp;
				tmp.x = x / a.x;
				tmp.y = y / a.y;
				tmp.z = z / a.z;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator+ (const vec& a) const
			{
				vec tmp;
				tmp.x = x + a.x;
				tmp.y = y + a.y;
				tmp.z = z + a.z;
				return tmp;
			}

			__device__ __SGOL_INLINE vec __SGOL_FASTCALL operator- (const vec& a) const
			{
				vec tmp;
				tmp.x = x - a.x;
				tmp.y = y - a.y;
				tmp.z = z - a.z;
				return tmp;
			}

			__device__ __SGOL_INLINE float& __SGOL_FASTCALL operator[](size_t i)
			{
				return m_Dim[i];
			}
		};

		typedef glm::vec2 vec2;
		typedef glm::vec3 vec3;
		typedef glm::vec4 vec4;

		typedef glm::mat2x3 mat2x3;
		typedef glm::mat4 mat4;
	}


#pragma pack(pop)

}
#if 0
template<typename T, uint8_t size>
__device__ inline SGOL::gmath::vec<T, size> operator*(float f, const SGOL::gmath::vec<T, size>& v)
{
	return v * f;
}

template<typename T, uint8_t size>
__device__ inline SGOL::gmath::vec<T, size> operator/(const SGOL::gmath::vec<T, size>& v, float f)
{
	return (1.0f / f) * v;
}

template<typename T, uint8_t size>
__device__ inline SGOL::gmath::vec<T, size> operator-(const SGOL::gmath::vec<T, size>& v)
{
	return -1 * v;
}
#endif

namespace SGOL {
	namespace gmath {

		template<typename T>
		static __host__ __device__ float length(const T& t)
		{
			return fastSqrt(length_squared(t));
		}

		template<typename T>
		static __host__ __device__ float length_squared(const T& t)
		{
			return t.x * t.x + t.y * t.y + t.z * t.z;
		}

		template<typename T>
		static __host__ __device__ float dot(const T& a, const T& b)
		{
			static_assert(false, "");
		}

		template<>
		static __host__ __device__ float dot(const vec2& a, const vec2& b)
		{
			vec2 tmp(a * b);
			return tmp.x + tmp.y;
		}

		template<>
		static __host__ __device__ float dot(const vec3& a, const vec3& b)
		{
			vec3 tmp(a * b);
			return tmp.x + tmp.y + tmp.z;
		}

		static __host__ __device__ vec3 fastNormalize(const vec3& v)
		{
			return v * fastInverseSqrt(dot(v, v));
		}

		static __host__ __device__ vec3 unit_vec(const vec3& v)
		{
			return v / length(v);
		}

		static __host__ __device__ float reflectness(float cosine, float ref_index)
		{
			float r0 = (1.0f - ref_index) / (1.0f + ref_index);
			r0 = r0 * r0;
			return r0 + (1.0f - r0) * (float)glm::pow(1.0f - cosine, 5);
		}

		static __host__ __device__ bool nearZero(const glm::vec3& v)
		{
			glm::vec3 tmp = glm::abs(v);
			const auto s = 1e-8;
			return (tmp.x < s) && (tmp.y < s) && (tmp.z < s);
		}

		static __host__ __device__ glm::vec3 refract(const glm::vec3& I, const glm::vec3& N, float eta)
		{
			float cos_theta = glm::min(glm::dot(-I, N), 1.0f);
			glm::vec3 r_out_perp = (I + cos_theta * N) * eta;
			glm::vec3 r_out_paralle = -fastInverseSqrt(glm::abs(1.0f - glm::dot(r_out_perp, r_out_perp))) * N;
			return r_out_perp + r_out_paralle;
		}

		typedef struct {
			uint64_t p;
			uint64_t q;
		} Nfactor;

		// Calculating factors "q" & "p" for given "N" and a guessed number "g"
		static __host__ __device__ uint8_t calc_factor(Nfactor* nf, uint64_t n, uint8_t g)
		{
			if (g < 1) return -1;
			// Calculating r for "g^r = mN + 1"
			uint64_t r = 1, sg = 1, p, q;
			for (; r < n; r++)
				if ((sg *= g) % n == 1)
					break;

			// calculating mq and mp 
			p = powl(g, r / 2) + 1;
			q = p - 2;

			// Crude Euclid's algurithm
			uint64_t c = n, tp;
			while (1)
			{
				if ((tp = (uint64_t)fmaxl(c, p) % (uint64_t)fminl(c, p)) == 0)
					break;
				p = tp;
			}

			// Saving the results for the given address
			(*nf).p = p;
			(*nf).q = n / p;

#ifdef CHECK_ON_ERROR
			return (*nf).p * (*nf).q != n;
#else
			return 0;
#endif
		}

		
		struct box
		{
			float width, height, x, y;

			__device__ box(float w, float h)
				: box(w, h, 0.0f, 0.0f)
			{}

			__device__ box(float w, float h, float x, float y)
				: width(w), height(h), x(x), y(y)
			{}

			__device__ float aspect() const { return width / height; }
		};


		constexpr static float infinity = std::numeric_limits<float>::infinity();

	}

}

#include "curand.h"
#include "curand_kernel.h"

namespace SGOL {

	class CuRandom
	{
	private:
		curandState_t* state = nullptr;
		size_t tid = 0;

	public:

		CuRandom()
		{}

		__device__ CuRandom(CuRandom& random)
			: state(random.state), tid(random.tid)
		{}

		__device__ CuRandom(curandState_t* state, size_t tid)
			: state(state), tid(tid)
		{}

		__device__ float random_float() const
		{
			return curand_uniform(&state[tid]);
		}

		__device__ float random_float(float min, float max) const
		{
			return min + (max - min) * random_float();
		}

		__device__ glm::vec2 randomVec2() const
		{
			return { random_float(), random_float()};
		}

		__device__ glm::vec2 randomVec2(float min, float max) const
		{
			return { random_float(min, max), random_float(min, max) };
		}

		__device__ glm::vec3 randomVec3() const
		{
			return { random_float(), random_float(), random_float() };
		}

		__device__ glm::vec3 randomVec3(float min, float max) const
		{
			return { random_float(min, max), random_float(min, max), random_float(min, max) };
		}

		__device__ glm::vec3 randomInUnitSphere() const
		{
			while (true)
			{
				glm::vec3 p = randomVec3(-1.0, 1.0);
				if (glm::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		__device__ gmath::vec3 randomInUnitDisk() const
		{
			while (true)
			{
				gmath::vec3 p = gmath::vec3(random_float(-1.0f, 1.0f), random_float(-1.0f, 1.0f), 0.0f);
				if (gmath::dot(p, p) >= 1.0f)
					continue;
				return p;
			}
		}

		__device__ CuRandom& operator()(size_t tid)
		{
			this->tid = tid;
			return *this;
		}

		__host__ __device__ void Clean()
		{
			cudaFree(state);
			state = nullptr;
			tid = 0;
		}

		__host__ __device__ operator curandState_t*()
		{
			return state;
		}

		__host__ __device__ curandState_t** operator&()
		{
			return &state;
		}

		__host__ __device__ operator bool()
		{
			return static_cast<bool>(state);
		}

		__device__ size_t GetTID() const { return tid; }

	};
}