#pragma once

#include "Math.h"

#include <cstdint>

/// <summary>
/// 
/// </summary>
namespace Utils {

	/// <summary>
	/// 
	/// </summary>
	namespace Color {

		/// <summary>
		/// 
		/// </summary>
		enum {
			/// <summary>
			/// 
			/// </summary>
			KeepColorChannel = 0xCC5
		};

		/// <summary>
		/// 
		/// </summary>
		/// <param name="buffer"></param>
		/// <param name="rgba"></param>
		static inline void RGBAtoColorChannels(uint8_t* buffer, uint32_t rgba)
		{
			buffer[0] = (rgba & 0xFF000000) >> 0x18;
			buffer[1] = (rgba & 0x00FF0000) >> 0x10;
			buffer[2] = (rgba & 0x0000FF00) >> 0x08;
			buffer[3] = (rgba & 0x000000FF);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ve"></param>
		/// <param name="rgba"></param>
		static inline void RGBAtoVec4(Math::Color4& ve, uint32_t rgba)
		{
			ve.a = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
			ve.b = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
			ve.g = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
			ve.r = (float)((rgba & 0x000000FF)) / 255.0f;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ve"></param>
		/// <param name="rgba"></param>
		static inline void RGBAtoVec3(Math::Color3& ve, uint32_t rgba)
		{
			//ve.a = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
			ve.b = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
			ve.g = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
			ve.r = (float)((rgba & 0x000000FF)) / 255.0f;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline Math::Color4 RGBAtoVec4(uint32_t rgba)
		{
			return {
				(float)((rgba & 0x000000FF)) / 255.0f,
				(float)((rgba & 0x0000FF00) >> 0x08) / 255.0f,
				(float)((rgba & 0x00FF0000) >> 0x10) / 255.0f,
				(float)((rgba & 0xFF000000) >> 0x18) / 255.0f };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline Math::Color3 RGBAtoVec3(uint32_t rgba)
		{
			return {
				(float)((rgba & 0x000000FF)) / 255.0f,
				(float)((rgba & 0x0000FF00) >> 0x08) / 255.0f,
				(float)((rgba & 0x00FF0000) >> 0x10) / 255.0f };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="r"></param>
		/// <param name="g"></param>
		/// <param name="b"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		static inline Math::Color4 RGBAtoVec4(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
		{
			return {
				(float)(r) / 255.0f,
				(float)(g) / 255.0f,
				(float)(b) / 255.0f,
				(float)(a) / 255.0f };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="r"></param>
		/// <param name="g"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		static inline Math::Color3 RGBAtoVec3(uint8_t r, uint8_t g, uint8_t b)
		{
			return {
				(float)(r) / 255.0f,
				(float)(g) / 255.0f,
				(float)(b) / 255.0f };
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="buffer"></param>
		/// <param name="rgba"></param>
		static inline void RGBAtoColorFloats(float* buffer, uint32_t rgba)
		{
			buffer[0] = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
			buffer[1] = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
			buffer[2] = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
			buffer[3] = (float)((rgba & 0x000000FF)) / 255.0f;
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline uint32_t RGBAtoBRGA(uint32_t rgba)
		{
			return (rgba & 0xFF000000) | (rgba & 0x00FF0000) >> 0x10 | (rgba & 0x0000FF00) << 0x08 | (rgba & 0x000000FF) << 0x08;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline uint32_t FlipRGBA(uint32_t rgba)
		{
			return (rgba & 0x000000FF) << 0x18 | (rgba & 0x0000FF00) << 0x08 | (rgba & 0x00FF0000) >> 0x08 | (rgba & 0xFF000000) >> 0x18;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline uint8_t GetAlphaChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0xFF000000) >> 0x18);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline uint8_t GetBlueChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0x00FF0000) >> 0x10);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline uint8_t GetGreenChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0x0000FF00) >> 0x08);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <returns></returns>
		static inline uint8_t GetRedChannel(uint32_t rgba)
		{
			return (uint8_t)((rgba & 0x000000FF));
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <param name="a"></param>
		static inline void SetAlphaChannel(uint32_t* rgba, uint8_t a)
		{
			(*rgba) = ((*rgba) & 0x00FFFFFF) | a << 0x18;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <param name="b"></param>
		static inline void SetBlueChannel(uint32_t* rgba, uint8_t b)
		{
			(*rgba) = ((*rgba) & 0xFF00FFFF) | b << 0x10;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <param name="g"></param>
		static inline void SetGreenChannel(uint32_t* rgba, uint8_t g)
		{
			(*rgba) = ((*rgba) & 0xFFFF00FF) | g << 0x08;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <param name="r"></param>
		static inline void SetRedChannel(uint32_t* rgba, uint8_t r)
		{
			(*rgba) = ((*rgba) & 0xFFFFFF00) | r;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="rgba"></param>
		/// <param name="r"></param>
		/// <param name="g"></param>
		/// <param name="b"></param>
		/// <param name="a"></param>
		static inline void SetColorChannel(uint32_t* rgba, uint16_t r = KeepColorChannel, uint16_t g = KeepColorChannel, uint16_t b = KeepColorChannel, uint16_t a = KeepColorChannel)
		{
			(*rgba) = a != KeepColorChannel ? ((*rgba) & 0x00FFFFFF) | a << 0x18 : (*rgba);
			(*rgba) = b != KeepColorChannel ? ((*rgba) & 0xFF00FFFF) | b << 0x10 : (*rgba);
			(*rgba) = g != KeepColorChannel ? ((*rgba) & 0xFFFF00FF) | g << 0x08 : (*rgba);
			(*rgba) = r != KeepColorChannel ? ((*rgba) & 0xFFFFFF00) | r : (*rgba);
		}


		/// <summary>
		/// 
		/// </summary>
		/// <param name="colors"></param>
		/// <returns></returns>
		static inline uint32_t ColorChannelsToRGBA(uint8_t* colors)
		{
			return colors[0] << 0x18 | colors[1] << 0x10 | colors[2] << 0x08 | colors[3];
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="r"></param>
		/// <param name="g"></param>
		/// <param name="b"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		static inline uint32_t ColorChannelsToRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF)
		{
			return a << 0x18 | b << 0x10 | g << 0x08 | r;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ve"></param>
		/// <returns></returns>
		static inline uint32_t Vec4ToRGBA(Math::Color4& ve)
		{
			return (uint8_t)(ve.a * 255.0f) << 0x18 | (uint8_t)(ve.b * 255.0f) << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ve"></param>
		/// <returns></returns>
		static inline uint32_t Vec3ToRGBA(Math::Color3& ve)
		{
			return 0xFF << 0x18 | (uint8_t)(ve.b * 255.0f) << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ve"></param>
		/// <returns></returns>
		static inline uint32_t Vec2ToRGBA(Math::Color2& ve)
		{
			return 0xFF << 0x18 | 0x00 << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ve"></param>
		/// <returns></returns>
		static inline uint32_t Vec1ToRGBA(Math::Color1& ve)
		{
			return 0xFF << 0x18 | 0x00 << 0x10 | 0x00 << 0x08 | (uint8_t)(ve.r * 255.0f);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="r"></param>
		/// <param name="g"></param>
		/// <param name="b"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		static inline uint32_t ColorFloatsToRGBA(float r, float g, float b, float a = 1.0f)
		{
			return (uint8_t)(a * 255.0f) << 0x18 | (uint8_t)(b * 255.0f) << 0x10 | (uint8_t)(g * 255.0f) << 0x08 | (uint8_t)(r * 255.0f);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="foreground"></param>
		/// <param name="background"></param>
		/// <returns></returns>
		static inline Math::Color4 Vec4ToRGBABlendColor(Math::Color4& foreground, Math::Color4& background)
		{
			return Math::Color4((Math::Color4(foreground) * (foreground.a)) + (Math::Color4(background) * (1.0f - foreground.a)));
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="foreground"></param>
		/// <param name="background"></param>
		/// <returns></returns>
		static inline uint32_t RGBABlendColor(uint32_t foreground, uint32_t background)
		{
			Math::Color4 foregroundVec;
			RGBAtoVec4(foregroundVec, foreground);
			//if (foregroundVec.a >= 1.0f)
				//return foreground;
			Math::Color4 backgroundVec;
			RGBAtoVec4(backgroundVec, background);
			return Vec4ToRGBA(Vec4ToRGBABlendColor(foregroundVec, backgroundVec));
		}
	}
}