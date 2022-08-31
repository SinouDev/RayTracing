#pragma once

#include "glm/glm.hpp"

#include <memory>

namespace ColorUtils {

	enum {
		KeepColorChannel = 0xCC5
	};

	static void RGBAtoColorChannels(uint8_t* buffer, uint32_t rgba)
	{
		buffer[0] = (rgba & 0xFF000000) >> 0x18;
		buffer[1] = (rgba & 0x00FF0000) >> 0x10;
		buffer[2] = (rgba & 0x0000FF00) >> 0x08;
		buffer[3] = (rgba & 0x000000FF);
	}

	static void RGBAtoVec4(glm::vec4& ve, uint32_t rgba)
	{
		ve.a = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
		ve.b = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
		ve.g = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
		ve.r = (float)((rgba & 0x000000FF)) / 255.0f;
	}

	static void RGBAtoVec3(glm::vec3& ve, uint32_t rgba)
	{
		//ve.a = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
		ve.b = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
		ve.g = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
		ve.r = (float)((rgba & 0x000000FF)) / 255.0f;
	}

	static void RGBAtoColorFloats(float* buffer, uint32_t rgba)
	{
		buffer[0] = (float)((rgba & 0xFF000000) >> 0x18) / 255.0f;
		buffer[1] = (float)((rgba & 0x00FF0000) >> 0x10) / 255.0f;
		buffer[2] = (float)((rgba & 0x0000FF00) >> 0x08) / 255.0f;
		buffer[3] = (float)((rgba & 0x000000FF)) / 255.0f;
	}


	static uint32_t RGBAtoBRGA(uint32_t rgba)
	{
		return (rgba & 0xFF000000) | (rgba & 0x00FF0000) >> 0x10 | (rgba & 0x0000FF00) << 0x08 | (rgba & 0x000000FF) << 0x08;
	}

	static uint32_t FlipRGBA(uint32_t rgba)
	{
		return (rgba & 0x000000FF) << 0x18 | (rgba & 0x0000FF00) << 0x08 | (rgba & 0x00FF0000) >> 0x08 | (rgba & 0xFF000000) >> 0x18;
	}

	static uint8_t GetAlphaChannel(uint32_t rgba)
	{
		return (uint8_t)((rgba & 0xFF000000) >> 0x18);
	}

	static uint8_t GetBlueChannel(uint32_t rgba)
	{
		return (uint8_t)((rgba & 0x00FF0000) >> 0x10);
	}

	static uint8_t GetGreenChannel(uint32_t rgba)
	{
		return (uint8_t)((rgba & 0x0000FF00) >> 0x08);
	}

	static uint8_t GetRedChannel(uint32_t rgba)
	{
		return (uint8_t)((rgba & 0x000000FF));
	}


	static void SetAlphaChannel(uint32_t* rgba, uint8_t a)
	{
		(*rgba) = ((*rgba) & 0x00FFFFFF) | a << 0x18;
	}

	static void SetBlueChannel(uint32_t* rgba, uint8_t b)
	{
		(*rgba) = ((*rgba) & 0xFF00FFFF) | b << 0x10;
	}

	static void SetGreenChannel(uint32_t* rgba, uint8_t g)
	{
		(*rgba) = ((*rgba) & 0xFFFF00FF) | g << 0x08;
	}

	static void SetRedChannel(uint32_t* rgba, uint8_t r)
	{
		(*rgba) = ((*rgba) & 0xFFFFFF00) | r;
	}

	static void SetColorChannel(uint32_t* rgba, uint16_t r = KeepColorChannel, uint16_t g = KeepColorChannel, uint16_t b = KeepColorChannel, uint16_t a = KeepColorChannel)
	{
		(*rgba) = a != KeepColorChannel ? ((*rgba) & 0x00FFFFFF) | a << 0x18 : (*rgba);
		(*rgba) = b != KeepColorChannel ? ((*rgba) & 0xFF00FFFF) | b << 0x10 : (*rgba);
		(*rgba) = g != KeepColorChannel ? ((*rgba) & 0xFFFF00FF) | g << 0x08 : (*rgba);
		(*rgba) = r != KeepColorChannel ? ((*rgba) & 0xFFFFFF00) | r : (*rgba);
	}


	static uint32_t ColorChannelsToRGBA(uint8_t* colors)
	{
		return colors[0] << 0x18 | colors[1] << 0x10 | colors[2] << 0x08 | colors[3];
	}

	static uint32_t ColorChannelsToRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF)
	{
		return a << 0x18 | b << 0x10 | g << 0x08 | r;
	}

	static uint32_t Vec4ToRGBA(glm::vec4& ve)
	{
		return (uint8_t)(ve.a * 255.0f) << 0x18 | (uint8_t)(ve.b * 255.0f) << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
	}

	static uint32_t Vec3ToRGBA(glm::vec3& ve)
	{
		return 0xFF << 0x18 | (uint8_t)(ve.b * 255.0f) << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
	}

	static uint32_t Vec2ToRGBA(glm::vec2& ve)
	{
		return 0xFF << 0x18 | 0x00 << 0x10 | (uint8_t)(ve.g * 255.0f) << 0x08 | (uint8_t)(ve.r * 255.0f);
	}

	static uint32_t Vec1ToRGBA(glm::vec1& ve)
	{
		return 0xFF << 0x18 | 0x00 << 0x10 | 0x00 << 0x08 | (uint8_t)(ve.r * 255.0f);
	}

	static uint32_t ColorFloatsToRGBA(float r, float g, float b, float a = 1.0f)
	{
		return (uint8_t)(a * 255.0f) << 0x18 | (uint8_t)(b * 255.0f) << 0x10 | (uint8_t)(g * 255.0f) << 0x08 | (uint8_t)(r * 255.0f);
	}

	static glm::vec4 Vec4ToRGBABlendColor(glm::vec4& foreground, glm::vec4& background)
	{
		return glm::vec4((glm::vec4(foreground) * (foreground.a)) + (glm::vec4(background) * (1.0f - foreground.a)));
	}

	static uint32_t RGBABlendColor(uint32_t foreground, uint32_t background)
	{
		glm::vec4 foregroundVec;
		RGBAtoVec4(foregroundVec, foreground);
		//if (foregroundVec.a >= 1.0f)
			//return foreground;
		glm::vec4 backgroundVec;
		RGBAtoVec4(backgroundVec, background);
		return Vec4ToRGBA(Vec4ToRGBABlendColor(foregroundVec, backgroundVec));
	}

};