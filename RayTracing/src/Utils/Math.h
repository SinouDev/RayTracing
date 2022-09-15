#pragma once

#define MATH_UTILS_DEFINED

#include "glm/glm.hpp"
#include "glm/gtx/fast_square_root.hpp"
#include "glm/gtc/constants.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <limits>
#include <cmath>

namespace Utils {


	/// <summary>
	/// Math wrapper library
	/// </summary>
	namespace Math {

		using Color1 = glm::vec1;
		using Color2 = glm::vec2;
		using Color3 = glm::vec3;
		using Color4 = glm::vec4;

		using Point2 = glm::vec2;
		using Point3 = glm::vec3;
		using Point4 = glm::vec4;

		using Vec2   = glm::vec2;
		using Vec3   = glm::vec3;
		using Vec4   = glm::vec4;

		using Mat2   = glm::mat2;
		using Mat3   = glm::mat3;
		using Mat4   = glm::mat4;

		using Mat2x2 = glm::mat2x2;
		using Mat2x3 = glm::mat2x3;
		using Mat2x4 = glm::mat2x4;

		using Mat3x2 = glm::mat3x2;
		using Mat3x3 = glm::mat3x3;
		using Mat3x4 = glm::mat3x4;

		using Mat4x2 = glm::mat4x2;
		using Mat4x3 = glm::mat4x3;
		using Mat4x4 = glm::mat4x4;

		using Quat   = glm::quat;

		using Coord  = glm::vec2;

		//using Dot = glm::dot;

		/// <summary>
		/// 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="t"></param>
		/// <returns></returns>
		template<typename T>
		static inline T UnitVec(const T& t)
		{

			//return glm::fastNormalize(t);
			return t / Q_Length(t);
		}

		template<typename T>
		static inline T Normalize(const T& t)
		{
			return glm::normalize(t);
		}

		template<typename T>
		static inline T Abs(const T& a)
		{
			return glm::abs(a);
		}

		static inline bool NearZero(const Vec3& vector)
		{
			Vec3 tmp = Abs(vector);
			const auto s = 1e-8;
			return (tmp.x < s) && (tmp.y < s) && (tmp.z < s);
		}

		static inline float Reflectness(float cosine, float ref_index)
		{
			float r0 = (1.0f - ref_index) / (1.0f + ref_index);
			r0 = r0 * r0;
			return static_cast<float>(r0 + (1.0f - r0) * glm::pow(1.0f - cosine, 5));
		}

		static inline float Sqrt(float n)
		{
			return glm::sqrt(n);
		}

		static inline float Rsqrt(float n)
		{
			return glm::inversesqrt(n);
		}

		template<typename T>
		static inline T Floor(const T& a)
		{
			return glm::floor(a);
		}

		static inline float Floor(float a)
		{
			return glm::floor(a);
		}

		template<typename T>
		static inline T Rotate(const Quat& t, const T& v)
		{
			return glm::rotate(t, v);
		}

		template<typename T>
		static inline float Dot(const T& x, const T& y)
		{
			return glm::dot(x, y);
		}

		static inline Quat AngleAxis(float angle, const Vec3& axis)
		{
			return glm::angleAxis(angle, axis);
		}

		template<typename T>
		static inline T Cross(const T& x, const T& y)
		{
			return glm::cross(x, y);
		}

		template<typename T>
		static inline T Min(const T& a, const T& b)
		{
			return glm::min(a, b);
		}

		template<typename T>
		static inline T Max(const T& a, const T& b)
		{
			return glm::max(a, b);
		}

		template<typename T>
		static inline T Log(const T& a)
		{
			return glm::log(a);
		}
		
		template<typename T>
		static inline float Sin(const T& a)
		{
			return glm::sin(a);
		}

		template<typename T>
		static inline float Cos(const T& a)
		{
			return glm::cos(a);
		}

		template<typename T>
		static inline float	Tan(const T& a)
		{
			return glm::tan(a);
		}

		static inline float Radians(float a)
		{
			return glm::radians(a);
		}

		static inline float Acos(float a)
		{
			return glm::acos(a);
		}

		static inline float Atan2(float yx, float xx)
		{
			return std::atan2(yx, xx);
		}

		template<typename T>
		static inline T Pi()
		{
			return glm::pi<T>();
		}

		template<typename T>
		static inline T TwoPi()
		{
			return glm::two_pi<T>();
		}

		template<typename T>
		static inline T Clamp(const T& x, const T& min, const T& max)
		{
			return glm::clamp(x, min, max);
		}

		template<typename T>
		static inline T Clamp(const T& x, float min, float max)
		{
			return glm::clamp(x, min, max);
		}

		static inline Mat4 PerspectiveFov(float fov, float width, float height, float near, float far)
		{
			return glm::perspectiveFov(fov, width, height, near, far);
		}

		template<typename T>
		static inline T Inverse(const T& a)
		{
			return glm::inverse(a);
		}

		static inline Mat4 LookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
		{
			return glm::lookAt(eye, center, up);
		}

		static inline Mat4 Translate(const Mat4& m, const Vec3& v)
		{
			return glm::translate(m, v);
		}

		static inline float __fastcall Q_Sqrt1(float n)
		{
			unsigned int i = *(unsigned int*)&n;
			// adjust bias
			i += 127 << 23;
			// approximation of square root
			i >>= 1;
			return *(float*)&i;
		}

		/// <summary>
		/// Watch this video to learn more -> https://youtu.be/p8u_k2LIZyo
		/// </summary>
		/// <param name="n"></param>
		/// <returns></returns>
		static inline float __fastcall Q_Rsqrt(float n)
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
			y = y * (1.5f - (x2 * y * y));
			y = y * (1.5f - (x2 * y * y));
			y = y * (1.5f - (x2 * y * y));

			return y;
		}

		static inline float __fastcall Q_Sqrt(float n)
		{
			return 1.0f / Q_Rsqrt(n);
		}

		static inline float __fastcall Q_Length(const Vec3& a)
		{
			return 1.0f / Q_Rsqrt(a.x * a.x + a.y * a.y + a.z * a.z);
		}

		static inline float __fastcall Q_Distance(const Vec3& a, const Vec3& b)
		{
			return Q_Length(b - a);
		}

		static inline Vec3 Refract(const Vec3& uv, const Vec3& n, float etai_over_etat)
		{
			float cos_theta = Min(Dot(-uv, n), 1.0f);
			Vec3 r_out_perp = (uv + cos_theta * n) * etai_over_etat;
			Vec3 r_out_paralle = -Q_Rsqrt(Abs(1.0f - Dot(r_out_perp, r_out_perp))) * n;
			return r_out_perp + r_out_paralle;
		}

		static inline Vec3 Reflect(const Vec3& v, const Vec3& n)
		{
			return v - 2.0f * Dot(v, n) * n;
		}

		/// <summary>
		/// 
		/// </summary>
		constexpr static float infinity = std::numeric_limits<float>::infinity();
	}
}