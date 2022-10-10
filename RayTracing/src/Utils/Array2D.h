#pragma once

#include "Utils.h"

namespace Utils {

	//
	// Simple 2D array class
	//
	// DataType: It is the underline data type
	// Note: Currently it's used for premetive types only
	template<typename DataType>
	class Array2D
	{
	public:

		__UTILS_INLINE Array2D(size_t width, size_t height)
			: m_Width(width), m_Height(height), m_Size(width * height)
		{
			m_Data = new DataType[m_Size];
		}

		__UTILS_INLINE Array2D()
		{}

		~Array2D()
		{
			delete[] m_Data;
		}

		__UTILS_INLINE operator const DataType* ()
		{
			return m_Data;
		}

		__UTILS_INLINE void __UTILS_FASTCALL SaveAndResize(size_t width, size_t height)
		{
			DataType* data = new DataType[width * height];

			for (size_t y = 0; y < m_Height; y++)
			{
				for (size_t x = 0; x < m_Width; x++)
				{
					size_t i = x + y * m_Width;
					data[i] = m_Data[i];
				}
			}

			delete[] m_Data;
			m_Data = data;
			m_Height = height;
			m_Width = width;
			m_Size = width * height;
		}

		// Resize will delete old data without saving it
		__UTILS_INLINE void __UTILS_FASTCALL Resize(size_t width, size_t height)
		{
			delete[] m_Data;

			//m_Height = height;
			//m_Width = width;
			//m_Size = width * height;
			//m_Data = new DataType[m_Size];

			m_Data = new DataType[m_Size = (m_Width = width) * (m_Height = height)];
		}

		__UTILS_INLINE void __UTILS_FASTCALL Fill(DataType fillValue)
		{
			for (size_t i = 0; i < m_Size; i++)
				m_Data[i] = fillValue;
		}
		
		template<typename T>
		__UTILS_INLINE T __UTILS_FASTCALL Width() const
		{
			return static_cast<T>(m_Width);
		}

		__UTILS_INLINE size_t __UTILS_FASTCALL Width() const
		{
			return m_Width;
		}
		
		template<typename T>
		__UTILS_INLINE T __UTILS_FASTCALL Height() const
		{
			return static_cast<T>(m_Height);
		}

		__UTILS_INLINE size_t __UTILS_FASTCALL Height() const
		{
			return m_Height;
		}
		
		template<typename T>
		__UTILS_INLINE T __UTILS_FASTCALL Size() const
		{
			return static_cast<T>(m_Size);
		}

		__UTILS_INLINE size_t __UTILS_FASTCALL Size() const
		{
			return m_Size;
		}

		template<typename T>
		__UTILS_INLINE T* __UTILS_FASTCALL Data()
		{
			return static_cast<T*>(m_Data);
		}

		__UTILS_INLINE DataType* __UTILS_FASTCALL Data()
		{
			return m_Data;
		}

		const __UTILS_INLINE DataType* __UTILS_FASTCALL Data() const
		{
			return m_Data;
		}

		__UTILS_INLINE Array2D<DataType>& __UTILS_FASTCALL Get()
		{
			return *this;
		}

		const __UTILS_INLINE Array2D<DataType>& __UTILS_FASTCALL Get() const
		{
			return *this;
		}

		__UTILS_INLINE DataType& __UTILS_FASTCALL operator()(size_t x, size_t y)
		{
			return m_Data[x + y * m_Width];
		}

		const __UTILS_INLINE DataType& __UTILS_FASTCALL operator()(size_t x, size_t y) const
		{
			return m_Data[x + y * m_Width];
		}

		__UTILS_INLINE DataType& __UTILS_FASTCALL operator[](size_t i)
		{
			return m_Data[i];
		}

		//template<typename T>
		//__UTILS_INLINE T& __UTILS_FASTCALL operator[](size_t i)
		//{
		//	return reinterpret_cast<T*>(m_Data)[i];
		//}

		const __UTILS_INLINE DataType& __UTILS_FASTCALL operator[](size_t i) const
		{
			return m_Data[i];
		}

	private:

		size_t m_Width = 0;
		size_t m_Height = 0;
		size_t m_Size = 0;
		DataType* m_Data = nullptr;
	};
}