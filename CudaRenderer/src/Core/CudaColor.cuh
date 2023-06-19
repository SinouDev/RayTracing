#pragma once

#include "cuda_runtime.h"

#include "gmath.cuh"

namespace SGOL {

	/**
	 * @brief Simple color struct that represents a color with RGBA channels.
	 *
	 * The Color struct provides a convenient way to work with colors by combining
	 * the red, green, blue, and alpha channels into a single struct. It supports
	 * various constructors and conversion operators for easy initialization and
	 * interoperability with other color representations such as glm::vec3 and glm::vec4.
	 * The color values are stored as floats ranging from 0.0 to 1.0.
	 *
	 * The struct provides arithmetic and assignment operators for basic color operations
	 * such as addition, subtraction, multiplication, and division. It also provides a
	 * Clamp() method to ensure that the color values are within the valid range.
	 */
	struct __builtin_align__(16) Color {

		// RGBA channels on union
		union {
			// 32-bit per color channel
			struct {
				float r, g, b, a;
			};

			float4 frgba;

			// Access value via glm::vec4 library
			glm::vec4 rgba;

			// Separate rgb and a channel for better control
			struct {
				glm::vec3 rgb;
				float a;
			};
		};

		/**
		 * @brief Constructs a Color object by copying another Color object.
		 * @param c The Color object to copy.
		 */
		__host__ __device__ Color(const Color& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = c.a;
		}

		/**
		 * @brief Constructs a Color object from a glm::vec3 object.
		 * @param c The glm::vec3 object representing RGB values.
		 */
		__host__ __device__ Color(const glm::vec3& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = 1.0f;
		}

		/**
		 * @brief Constructs a Color object from a glm::vec4 object.
		 * @param c The glm::vec4 object representing RGBA values.
		 */
		__host__ __device__ Color(const glm::vec4& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = c.a;
		}

		__device__ Color(const float4& c)
		{
			frgba = c;
			/*
			r = c.x;
			g = c.y;
			b = c.z;
			a = c.w;
			*/
		}

		/**
		 * @brief Constructs a Color object with all channels set to zero.
		 */
		__host__ __device__ Color()
			: Color(0.0f)
		{}

		/**
		 * @brief Constructs a Color object with all channels set to the same value.
		 * @param c The value to set for all channels.
		 */
		__host__ __device__ Color(float c)
			: Color(c, c, c, c)
		{}

		/**
		 * @brief Constructs a Color object from individual RGBA channel values in the range [0, 255].
		 * @param r The red channel value.
		 * @param g The green channel value.
		 * @param b The blue channel value.
		 * @param a The alpha channel value.
		 */
		__host__ __device__ __SGOL_INLINE Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
			: Color(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f)
		{}

		/**
		 * @brief Constructs a Color object from individual RGBA channel values in the range [0.0, 1.0].
		 * @param r The red channel value.
		 * @param g The green channel value.
		 * @param b The blue channel value.
		 * @param a The alpha channel value (default is 1.0).
		 */
		__host__ __device__ __SGOL_INLINE Color(float r, float g, float b, float a = 1.0f)
			: rgba(r, g, b, a) //r(r), g(g), b(b), a(a)
		{}

		/**
		 * @brief Converts the Color object to a 32-bit unsigned integer representation.
		 * @return The 32-bit unsigned integer representation of the color.
		 */
		constexpr explicit __host__ __device__ __SGOL_INLINE operator uint32_t()
		{
			uint32_t res = 0;
			res |= static_cast<uint8_t>(r * 255.0f) | static_cast<uint8_t>(g * 255.0f) << 8 | static_cast<uint8_t>(b * 255.0f) << 16 | static_cast<uint8_t>(a * 255.0f) << 24;
			return res;
		}

		/**
		 * @brief Converts the Color object to a glm::vec4 object.
		 * @return The glm::vec4 representation of the color.
		 */
		constexpr __host__ __device__ __SGOL_INLINE operator glm::vec4&()
		{
			return rgba;
		}

		/**
		 * @brief Converts the Color object to a const glm::vec4 object.
		 * @return The const glm::vec4 representation of the color.
		 */
		constexpr __host__ __device__ __SGOL_INLINE operator const glm::vec4&() const
		{
			return rgba;
		}

		/**
		 * @brief Converts the Color object to a glm::vec3 object.
		 * @return The glm::vec3 representation of the color.
		 */
		constexpr __host__ __device__ __SGOL_INLINE operator glm::vec3&()
		{
			return rgb;
		}

		/**
		 * @brief Converts the Color object to a const glm::vec3 object.
		 * @return The const glm::vec3 representation of the color.
		 */
		constexpr __host__ __device__ __SGOL_INLINE operator const glm::vec3&() const
		{
			return rgb;
		}

		/**
		 * @brief Constructs a Color object from a 32-bit unsigned integer representation.
		 * @param c The 32-bit unsigned integer representation of the color.
		 */
		__host__ __device__ Color(uint32_t c)
		{
			a = static_cast<float>((c & 0xFF000000) >> 24) / 255.0f;
			b = static_cast<float>((c & 0x00FF0000) >> 16) / 255.0f;
			g = static_cast<float>((c & 0x0000FF00) >> 8) / 255.0f;
			r = static_cast<float>(c & 0x000000FF) / 255.0f;
		}

		/**
		 * @brief Clamps the color values to the range [0.0, 1.0].
		 * @return The clamped Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& Clamp()
		{
			//r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
			//g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
			//b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
			//a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
			ClampLeft();
			ClampRight();
			return *this;
		}

		/**
		 * @brief Clamps the left side of the color values to the range [0.0, ∞).
		 * @return The clamped Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& ClampLeft()
		{
			r = r < 0.0f ? 0.0f : r;
			g = g < 0.0f ? 0.0f : g;
			b = b < 0.0f ? 0.0f : b;
			a = a < 0.0f ? 0.0f : a;
			return *this;
		}

		/**
		 * @brief Clamps the right side of the color values to the range (∞, 1.0].
		 * @return The clamped Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& ClampRight()
		{
			r = r > 1.0f ? 1.0f : r;
			g = g > 1.0f ? 1.0f : g;
			b = b > 1.0f ? 1.0f : b;
			a = a > 1.0f ? 1.0f : a;
			return *this;
		}

		/**
		 * @brief Assigns a Color object to another Color object.
		 * @param c The Color object to assign.
		 * @return The assigned Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& operator=(const Color& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = c.a;
			return *this;
		}

		/**
		 * @brief Adds two Color objects.
		 * @param c The Color object to add.
		 * @return The result of the addition.
		 */
		__host__ __device__ __SGOL_INLINE Color operator+(const Color& c) const
		{
			Color tmp;
			tmp.r = r + c.r;
			tmp.g = g + c.g;
			tmp.b = b + c.b;
			tmp.a = a + c.a;
			return tmp;
		}

		/**
		 * @brief Subtracts two Color objects.
		 * @param c The Color object to subtract.
		 * @return The result of the subtraction.
		 */
		__host__ __device__ __SGOL_INLINE Color operator-(const Color& c) const
		{
			Color tmp;
			tmp.r = r - c.r;
			tmp.g = g - c.g;
			tmp.b = b - c.b;
			tmp.a = a - c.a;
			return tmp;
		}

		/**
		 * @brief Divides two Color objects.
		 * @param c The Color object to divide by.
		 * @return The result of the division.
		 */
		__host__ __device__ __SGOL_INLINE Color operator/(const Color& c) const
		{
			Color tmp;
			tmp.r = r / c.r;
			tmp.g = g / c.g;
			tmp.b = b / c.b;
			tmp.a = a / c.a;
			return tmp;
		}

		/**
		 * @brief Multiplies two Color objects.
		 * @param c The Color object to multiply.
		 * @return The result of the multiplication.
		 */
		__host__ __device__ __SGOL_INLINE Color operator*(const Color& c) const
		{
			Color tmp;
			tmp.r = r * c.r;
			tmp.g = g * c.g;
			tmp.b = b * c.b;
			tmp.a = a * c.a;
			return tmp;
		}

		/**
		 * @brief Adds the components of another Color object to this Color object.
		 *
		 * @param c The Color object to add.
		 * @return A reference to the modified Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& operator+=(const Color& c)
		{
			this->r += c.r;
			this->g += c.g;
			this->b += c.b;
			this->a += c.a;
			return *this;
		}

		/**
		 * @brief Subtracts the components of another Color object from this Color object.
		 *
		 * @param c The Color object to subtract.
		 * @return A reference to the modified Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& operator-=(const Color& c)
		{
			this->r -= c.r;
			this->g -= c.g;
			this->b -= c.b;
			this->a -= c.a;
			return *this;
		}

		/**
		 * @brief Divides the components of this Color object by the corresponding components of another Color object.
		 *
		 * @param c The Color object to divide by.
		 * @return A reference to the modified Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& operator/=(const Color& c)
		{
			this->r /= c.r;
			this->g /= c.g;
			this->b /= c.b;
			this->a /= c.a;
			return *this;
		}

		/**
		 * @brief Multiplies the components of this Color object by the corresponding components of another Color object.
		 *
		 * @param c The Color object to multiply by.
		 * @return A reference to the modified Color object.
		 */
		__host__ __device__ __SGOL_INLINE Color& operator*=(const Color& c)
		{
			this->r *= c.r;
			this->g *= c.g;
			this->b *= c.b;
			this->a *= c.a;
			return *this;
		}

		constexpr __device__ __SGOL_INLINE operator float4() noexcept
		{
			return frgba;
		}

	};

} // namespace SGOL


#if 0
#pragma once

#include "cuda_runtime.h"

#include "gmath.cuh"

namespace SGOL {

	/// <summary>
	/// Simple color struct that intersects between raw data and color channels
	/// </summary>
	struct Color {

		// RGBA channels on union
		union {
			// 32 bit per color channel 
			struct {
				float r, g, b, a;
			};

			// access value via glm::vec4 library
			glm::vec4 rgba;

			// Seperate rgb and a channel for better control
			struct {
				glm::vec3 rgb;
				float a;
			};
		};

		__host__ __device__ Color(const Color& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = c.a;
		}

		__host__ __device__ Color(const glm::vec3& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = 1.0f;
		}

		__host__ __device__ Color(const glm::vec4& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = c.a;
		}

		__host__ __device__ Color()
			: Color(0.0f)
		{}

		__host__ __device__ Color(float c)
			: Color(c, c, c, c)
		{}

		__host__ __device__ inline Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
			: Color(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f)
		{}

		__host__ __device__ inline Color(float r, float g, float b, float a = 1.0f)
			: rgba(r, g, b, a) //r(r), g(g), b(b), a(a)
		{}

		explicit __host__ __device__ inline operator uint32_t ()
		{
			uint32_t res = 0;
			res |= static_cast<uint8_t>(r * 255.0f) | static_cast<uint8_t>(g * 255.0f) << 8 | static_cast<uint8_t>(b * 255.0f) << 16 | static_cast<uint8_t>(a * 255.0f) << 24;
			return res;
		}

		__host__ __device__ inline operator glm::vec4& ()
		{
			return rgba;
		}

		__host__ __device__ inline operator const glm::vec4& () const
		{
			return rgba;
		}

		__host__ __device__ inline operator glm::vec3& ()
		{
			return rgb;
		}

		__host__ __device__ inline operator const glm::vec3& () const
		{
			return rgb;
		}

		__host__ __device__ inline Color(uint32_t c)
		{
			a = static_cast<float>((c & 0xFF000000) >> 24) / 255.0f;
			b = static_cast<float>((c & 0x00FF0000) >> 16) / 255.0f;
			g = static_cast<float>((c & 0x0000FF00) >> 8) / 255.0f;
			r = static_cast<float>(c & 0x000000FF) / 255.0f;
		}

		__host__ __device__ Color& Clamp()
		{
			r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
			g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
			b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
			a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
			return *this;
		}

		__host__ __device__ Color& operator= (const Color& c)
		{
			r = c.r;
			g = c.g;
			b = c.b;
			a = c.a;
			return *this;
		}

		__host__ __device__ Color operator+ (const Color& c) const
		{
			Color tmp;
			tmp.r = r + c.r;
			tmp.g = g + c.g;
			tmp.b = b + c.b;
			tmp.a = a + c.a;
			return tmp;
		}

		__host__ __device__ Color operator- (const Color& c) const
		{
			Color tmp;
			tmp.r = r - c.r;
			tmp.g = g - c.g;
			tmp.b = b - c.b;
			tmp.a = a - c.a;
			return tmp;
		}

		__host__ __device__ Color operator/ (const Color& c) const
		{
			Color tmp;
			tmp.r = r / c.r;
			tmp.g = g / c.g;
			tmp.b = b / c.b;
			tmp.a = a / c.a;
			return tmp;
		}

		__host__ __device__ Color operator* (const Color& c) const
		{
			Color tmp;
			tmp.r = r * c.r;
			tmp.g = g * c.g;
			tmp.b = b * c.b;
			tmp.a = a * c.a;
			return tmp;
		}


		__host__ __device__ Color& operator+= (const Color& c)
		{
			this->r += c.r;
			this->g += c.g;
			this->b += c.b;
			this->a += c.a;
			return *this;
		}

		__host__ __device__ Color& operator-= (const Color& c)
		{
			this->r -= c.r;
			this->g -= c.g;
			this->b -= c.b;
			this->a -= c.a;
			return *this;
		}

		__host__ __device__ Color& operator/= (const Color& c)
		{
			this->r /= c.r;
			this->g /= c.g;
			this->b /= c.b;
			this->a /= c.a;
			return *this;
		}

		__host__ __device__ Color& operator*= (const Color& c)
		{
			this->r *= c.r;
			this->g *= c.g;
			this->b *= c.b;
			this->a *= c.a;
			return *this;
		}

	};
}
#endif

/**
 * @brief Adds a float value to a Color object.
 * @param f The float value to add.
 * @param c The Color object to add to.
 * @return The result of the addition.
 */
__host__ __device__ inline SGOL::Color operator+(float f, const SGOL::Color& c)
{
	return c + f;
}

/**
 * @brief Subtracts a float value from a Color object.
 * @param f The float value to subtract.
 * @param c The Color object to subtract from.
 * @return The result of the subtraction.
 */
__host__ __device__ inline SGOL::Color operator-(float f, const SGOL::Color& c)
{
	return c - f;
}

/**
 * @brief Divides a float value by a Color object.
 * @param f The float value to divide.
 * @param c The Color object to divide by.
 * @return The result of the division.
 */
__host__ __device__ inline SGOL::Color operator/(float f, const SGOL::Color& c)
{
	return c / f;
}

/**
 * @brief Multiplies a float value with a Color object.
 * @param f The float value to multiply.
 * @param c The Color object to multiply with.
 * @return The result of the multiplication.
 */
__host__ __device__ inline SGOL::Color operator*(float f, const SGOL::Color& c)
{
	return c * f;
}

__device__ inline float4& operator+=(float4& f, float ff)
{
	f.x += ff;
	f.y += ff;
	f.z += ff;
	f.w += ff;
	return f;
}

__device__ inline float4& operator-=(float4& f, float ff)
{
	f.x -= ff;
	f.y -= ff;
	f.z -= ff;
	f.w -= ff;
	return f;
}

__device__ inline float4& operator/=(float4& f, float ff)
{
	f.x /= ff;
	f.y /= ff;
	f.z /= ff;
	f.w /= ff;
	return f;
}

__device__ inline float4& operator*=(float4& f, float ff)
{
	f.x *= ff;
	f.y *= ff;
	f.z *= ff;
	f.w *= ff;
	return f;
}

__device__ inline float4& operator+=(float4& f, const float4& ff)
{
	f.x += ff.x;
	f.y += ff.y;
	f.z += ff.z;
	f.w += ff.w;
	return f;
}

__device__ inline float4& operator-=(float4& f, const float4& ff)
{
	f.x -= ff.x;
	f.y -= ff.y;
	f.z -= ff.z;
	f.w -= ff.w;
	return f;
}

__device__ inline float4& operator/=(float4& f, const float4& ff)
{
	f.x /= ff.x;
	f.y /= ff.y;
	f.z /= ff.z;
	f.w /= ff.w;
	return f;
}

__device__ inline float4& operator*=(float4& f, const float4& ff)
{
	f.x *= ff.x;
	f.y *= ff.y;
	f.z *= ff.z;
	f.w *= ff.w;
	return f;
}