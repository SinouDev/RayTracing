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

#include "Utils.h"

//extern "C" float __UTILS_FASTCALL Q_Sqrt1(float n);

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
		static __UTILS_INLINE T __UTILS_FASTCALL UnitVec(const T& t)
		{
			return t / Q_Length(t);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Normalize(const T& t)
		{
			return glm::normalize(t);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Abs(const T& a)
		{
			return glm::abs(a);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Pow(float xx, float yx)
		{
			return glm::pow(xx, yx);
		}

		static __UTILS_INLINE bool __UTILS_FASTCALL NearZero(const Vec3& vector)
		{
			Vec3 tmp = Abs(vector);
			const auto s = 1e-8;
			return (tmp.x < s) && (tmp.y < s) && (tmp.z < s);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Reflectness(float cosine, float ref_index)
		{
			float r0 = (1.0f - ref_index) / (1.0f + ref_index);
			r0 = r0 * r0;
			return r0 + (1.0f - r0) * Pow(1.0f - cosine, 5);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Sqrt(float n)
		{
			return glm::sqrt(n);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Rsqrt(float n)
		{
			return glm::inversesqrt(n);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Floor(const T& a)
		{
			return glm::floor(a);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Floor(float a)
		{
			return glm::floor(a);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Rotate(const Quat& t, const T& v)
		{
			return glm::rotate(t, v);
		}

		template<typename T>
		static __UTILS_INLINE float __UTILS_FASTCALL Dot(const T& x, const T& y)
		{
			return glm::dot(x, y);
		}

		static __UTILS_INLINE Quat __UTILS_FASTCALL AngleAxis(float angle, const Vec3& axis)
		{
			return glm::angleAxis(angle, axis);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Cross(const T& x, const T& y)
		{
			return glm::cross(x, y);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Min(const T& a, const T& b)
		{
			return glm::min(a, b);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Max(const T& a, const T& b)
		{
			return glm::max(a, b);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Log(const T& a)
		{
			return glm::log(a);
		}

		template<typename T>
		static __UTILS_INLINE float __UTILS_FASTCALL Sin(const T& a)
		{
			return glm::sin(a);
		}

		template<typename T>
		static __UTILS_INLINE float __UTILS_FASTCALL Cos(const T& a)
		{
			return glm::cos(a);
		}

		template<typename T>
		static __UTILS_INLINE float __UTILS_FASTCALL Tan(const T& a)
		{
			return glm::tan(a);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Radians(float a)
		{
			return glm::radians(a);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Degrees(float a)
		{
			return glm::degrees(a);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Acos(float a)
		{
			return glm::acos(a);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Atan2(float yx, float xx)
		{
			return std::atan2(yx, xx);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Pi()
		{
			return glm::pi<T>();
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL TwoPi()
		{
			return glm::two_pi<T>();
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Clamp(const T& x, const T& min, const T& max)
		{
			return glm::clamp(x, min, max);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Clamp(const T& x, float min, float max)
		{
			return glm::clamp(x, min, max);
		}

		static __UTILS_INLINE Mat4 __UTILS_FASTCALL PerspectiveFov(float fov, float width, float height, float near, float far)
		{
			return glm::perspectiveFov(fov, width, height, near, far);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Inverse(const T& a)
		{
			return glm::inverse(a);
		}

		static __UTILS_INLINE Mat4 __UTILS_FASTCALL LookAt(const Vec3& eye, const Vec3& center, const Vec3& up)
		{
			return glm::lookAt(eye, center, up);
		}

		static __UTILS_INLINE Mat4 __UTILS_FASTCALL Translate(const Mat4& m, const Vec3& v)
		{
			return glm::translate(m, v);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Q_Sqrt1(float n)
		{
			//unsigned int i = *(unsigned int*)&n;
			//// adjust bias
			//i += 127 << 23;
			//// approximation of square root
			//i >>= 1;
			//return *(float*)&i;
		}

		/// <summary>
		/// Watch this video to learn more -> https://youtu.be/p8u_k2LIZyo
		/// </summary>
		/// <param name="n"></param>
		/// <returns></returns>
		static __UTILS_INLINE float __UTILS_FASTCALL Q_Rsqrt(float n)
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

			return y;
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Q_Sqrt(float n)
		{
			return 1.0f / Q_Rsqrt(n);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Q_Length(const Vec3& a)
		{
			return Q_Sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
		}

		static __UTILS_INLINE float __UTILS_FASTCALL Q_Distance(const Vec3& a, const Vec3& b)
		{
			return Q_Length(b - a);
		}

		template<typename T>
		static __UTILS_INLINE T __UTILS_FASTCALL Q_Normalize(const T& t)
		{
			return t * Q_Rsqrt(Dot(t, t));
		}

		static __UTILS_INLINE Vec3 __UTILS_FASTCALL Refract(const Vec3& uv, const Vec3& n, float etai_over_etat)
		{
			float cos_theta = Min(Dot(-uv, n), 1.0f);
			Vec3 r_out_perp = (uv + cos_theta * n) * etai_over_etat;
			Vec3 r_out_paralle = -Q_Rsqrt(Abs(1.0f - Dot(r_out_perp, r_out_perp))) * n;
			return r_out_perp + r_out_paralle;
		}

		static __UTILS_INLINE Vec3 __UTILS_FASTCALL Reflect(const Vec3& v, const Vec3& n)
		{
			return v - 2.0f * Dot(v, n) * n;
		}

		/// <summary>
		/// 
		/// </summary>
		constexpr static float infinity = std::numeric_limits<float>::infinity();
	}
}