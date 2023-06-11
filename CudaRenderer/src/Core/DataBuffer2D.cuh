#pragma once

#include "cuda_runtime.h"

#include <stdint.h>

#include "SGOL/Array2D.hpp"
#include "glm/glm.hpp"

class CudaRenderer;

namespace SGOL {

    /**
     * @brief Represents a 2D data buffer for CUDA operations.
     * 
     * The DataBuffer2D class provides a container for storing and manipulating 2D data in CUDA operations. It is
     * designed to work with the CudaRenderer class for efficient GPU-based rendering.
     *
     * @tparam _Ty The type of elements stored in the buffer.
     */
    template<typename _Ty>
    class DataBuffer2D : public Array2D<_Ty>
    {

    public:

        /**
         * @brief Flag indicating that the buffer is a copy.
         */
        static constexpr char FLAG_IS_COPY = 0x1;

        /**
         * @brief Flag indicating that the buffer is busy.
         */
        static constexpr char FLAG_IS_BUSY = 0x2;

        /**
         * @brief Constructs a DataBuffer2D object with the specified dimensions.
         *
         * This constructor initializes a new DataBuffer2D object with the given width and height.
         *
         * @param width The width of the buffer.
         * @param height The height of the buffer.
         */
        __host__ DataBuffer2D(size_t width, size_t height)
        {
            m_UserHandle = true;
            Allocate(width, height);
        }

        /**
         * @brief Default constructor.
         *
         * This constructor creates an empty DataBuffer2D object without initializing any storage.
         */
        __device__ __host__ DataBuffer2D()
        {
            m_UserHandle = true;
        }

        /**
         * @brief Destructor.
         *
         * The destructor releases the allocated memory if the buffer is not a copy.
         */
        virtual __device__ __host__ ~DataBuffer2D() override
        {
            if (!(m_Flags & FLAG_IS_COPY))
                Free();
        }

        /**
         * @brief Copy constructor.
         *
         * This constructor creates a copy of the specified DataBuffer2D object.
         *
         * @param cpy The DataBuffer2D object to copy.
         */
        __device__ __host__ DataBuffer2D(const DataBuffer2D& cpy)
        {
            m_Data = cpy.m_Data;
            m_UserHandle = cpy.m_UserHandle;
            m_Width = cpy.m_Width;
            m_Height = cpy.m_Height;
            m_Size = cpy.m_Size;
            m_Flags = cpy.m_Flags;

            m_Flags = (m_Flags & ~FLAG_IS_COPY) | 1;

        }

        __device__ __host__ __SGOL_INLINE _Ty& __SGOL_FASTCALL operator()(size_t x, size_t y)
        {
            return m_Data[x + y * m_Width];
        }

        __device__ __host__ __SGOL_INLINE const _Ty& __SGOL_FASTCALL operator()(size_t x, size_t y) const
        {
            return m_Data[x + y * m_Width];
        }

        __device__ __host__ __SGOL_INLINE _Ty& __SGOL_FASTCALL operator()(const glm::vec2& coord)
        {
            return m_Data[(size_t)(coord.x + coord.y * m_Width)];
        }

        __device__ __host__ __SGOL_INLINE const _Ty& __SGOL_FASTCALL operator()(const glm::vec2& coord) const
        {
            return m_Data[(size_t)(coord.x + coord.y * m_Width)];
        }

        /**
         * @brief Access operator.
         *
         * This operator allows accessing the elements of the buffer using the [] operator.
         *
         * @param i The index of the element to access.
         * @return A reference to the element at the specified index.
         */
        __SGOL_INLINE __device__ __host__ _Ty& operator[](size_t i)
        {
            return m_Data[i];
        }

        /**
         * @brief Const access operator.
         *
         * This operator allows accessing the elements of the buffer using the [] operator when the buffer is const.
         *
         * @param i The index of the element to access.
         * @return A const reference to the element at the specified index.
         */
        __SGOL_INLINE __device__ __host__ const _Ty& operator[](size_t i) const
        {
            return m_Data[i];
        }

        /**
         * @brief Retrieves the coordinates of a point in the buffer.
         *
         * This method calculates and returns the (x, y) coordinates of the element at the specified index in the buffer.
         *
         * @param i The index of the element.
         * @return The (x, y) coordinates of the element.
         */
        __SGOL_INLINE __device__ __host__ glm::vec2 Point(size_t i) const
        {
            return glm::vec2{ (size_t)(i% m_Width), (size_t)(i / m_Width) };
        }

        /**
         * @brief Retrieves the dimensions of the buffer.
         *
         * This method returns the width and height of the buffer as a glm::vec2 object.
         *
         * @return The dimensions of the buffer.
         */
        __SGOL_INLINE __device__ __host__ glm::vec2 Dimentions() const
        {
            return glm::vec2{ m_Width, m_Height };
        }

        /**
         * @brief Resizes the buffer to the specified dimensions.
         *
         * This method resizes the buffer to the specified width and height.
         *
         * @param width The new width of the buffer.
         * @param height The new height of the buffer.
         */
        virtual __SGOL_INLINE void __SGOL_FASTCALL Resize(size_t width, size_t height) override
        {
            SetBusyFlag(true);

            Free();

            Allocate(width, height);

            SetBusyFlag(false);
        }

        /**
         * @brief Frees the memory allocated for the buffer.
         *
         * This method frees the memory allocated for the buffer on the device.
         */
        __SGOL_INLINE __host__ void Free()
        {
            if (!(m_Flags & FLAG_IS_COPY))
                cudaFree(m_Data);
        }

        /**
         * @brief Checks if the buffer is busy.
         *
         * This method checks if the buffer is currently being used (busy).
         *
         * @return True if the buffer is busy, false otherwise.
         */
        __SGOL_INLINE bool IsBusy() { return m_Flags & FLAG_IS_BUSY; }

        /**
         * @brief Sets the number of threads per block for CUDA operations.
         *
         * This method sets the number of threads per block to be used in CUDA operations.
         *
         * @param x The number of threads in the x dimension.
         * @param y The number of threads in the y dimension.
         */
        __SGOL_INLINE void SetThreadsPerBlock(size_t x, size_t y) { m_ThreadsPerBlock = dim3(x, y); }

        /**
         * @brief Sets the number of threads per block for CUDA operations.
         *
         * This method sets the number of threads per block to be used in CUDA operations.
         *
         * @param threads The number of threads per block as a dim3 object.
         */
        __SGOL_INLINE void SetThreadsPerBlock(dim3 threads) { m_ThreadsPerBlock = threads; }

        /**
         * @brief Retrieves the number of threads per block for CUDA operations.
         *
         * This method returns the number of threads per block to be used in CUDA operations.
         *
         * @return The number of threads per block as a dim3 object.
         */
        __SGOL_INLINE dim3 ThreadsPerBlock() { return m_ThreadsPerBlock; }

        /**
         * @brief Retrieves the number of blocks for CUDA operations.
         *
         * This method returns the number of blocks to be used in CUDA operations based on the buffer dimensions and the number of threads per block.
         *
         * @return The number of blocks as a dim3 object.
         */
        __SGOL_INLINE dim3 NumBlocks() { return m_NumBlocks; }

    private:

        friend class CudaRenderer;

        /**
         * @brief Allocates memory for the buffer on the device.
         *
         * This method allocates memory for the buffer on the device using cudaMallocManaged.
         *
         * @param width The width of the buffer.
         * @param height The height of the buffer.
         */
        __SGOL_INLINE __host__ void Allocate(size_t width, size_t height)
        {
            m_Width = width;
            m_Height = height;
            m_Size = width * height;

            cudaError_t err = cudaMallocManaged(&m_Data, m_Size * sizeof(Type));

            if (err != cudaError::cudaSuccess)
            {
                printf("Error while allocating memory on the device with size of %llu, with error code: 0x%016X\n", m_Size * sizeof(Type), err);
            }

            m_NumBlocks = dim3(m_Width / m_ThreadsPerBlock.x, m_Height / m_ThreadsPerBlock.y);
        }

        /**
         * @brief Sets the busy flag of the buffer.
         *
         * This method sets the busy flag of the buffer to indicate whether it is currently being used.
         *
         * @param busy True if the buffer is busy, false otherwise.
         */
        inline void SetBusyFlag(bool busy) { m_Flags = (m_Flags & ~FLAG_IS_BUSY) | ((busy ? 1 : 0) << 1); }

    private:

        uint8_t m_Flags = 0x0;   ///< Flags indicating the status of the buffer.
        dim3 m_ThreadsPerBlock, m_NumBlocks;   ///< Number of threads per block and number of blocks for CUDA operations.
    };

}


#if 0
#pragma once

#include "cuda_runtime.h"

#include <stdint.h>

#include "SGOL/Array2D.hpp"
#include "glm/glm.hpp"

class CudaRenderer;

namespace SGOL {

	template<typename _Ty>
	class DataBuffer2D : public Array2D<_Ty>
	{

	public:

		static constexpr char FLAG_IS_COPY = 0x1;
		static constexpr char FLAG_IS_BUSY = 0x2;

		__host__ DataBuffer2D(size_t width, size_t height)
		{
			m_UserHandle = true;
			Allocate(width, height);
		}

		__device__ __host__ DataBuffer2D()
		{
			m_UserHandle = true;
		}

		virtual __device__ __host__ ~DataBuffer2D() override
		{
			if (!(m_Flags & FLAG_IS_COPY))
				Free();
		}

		__device__ __host__ DataBuffer2D(const DataBuffer2D& cpy)
		{
			m_Data = cpy.m_Data;
			m_UserHandle = cpy.m_UserHandle;
			m_Width = cpy.m_Width;
			m_Height = cpy.m_Height;
			m_Size = cpy.m_Size;
			m_Flags = cpy.m_Flags;

			m_Flags = (m_Flags & ~FLAG_IS_COPY) | 1;

		}

		__SGOL_INLINE __device__ __host__ _Ty& operator[](size_t i)
		{
			return m_Data[i];
		}

		__SGOL_INLINE __device__ __host__ const _Ty& operator[](size_t i) const
		{
			return m_Data[i];
		}

		__SGOL_INLINE __device__ __host__ glm::vec2 Point(size_t i) const
		{
			return glm::vec2{ (size_t)(i % m_Width), (size_t)(i / m_Width) };
		}

		__SGOL_INLINE __device__ __host__ glm::vec2 Dimentions() const
		{
			return glm::vec2{ m_Width, m_Height };
		}

		virtual __SGOL_INLINE void __SGOL_FASTCALL Resize(size_t width, size_t height) override
		{
			SetBusyFlag(true);

			Free();

			Allocate(width, height);

			SetBusyFlag(false);
		}

		__SGOL_INLINE __host__ void Free()
		{
			if (!(m_Flags & FLAG_IS_COPY))
				cudaFree(m_Data);
		}

		__SGOL_INLINE bool IsBusy() { return m_Flags & FLAG_IS_BUSY; }

		__SGOL_INLINE void SetThreadsPerBlock(size_t x, size_t y) { m_ThreadsPerBlock = dim3(x, y); }
		__SGOL_INLINE void SetThreadsPerBlock(dim3 threads) { m_ThreadsPerBlock = threads; }

		__SGOL_INLINE dim3 ThreadsPerBlock() { return m_ThreadsPerBlock; }
		__SGOL_INLINE dim3 NumBlocks() { return m_NumBlocks; }

	private:

		friend class CudaRenderer;

		__SGOL_INLINE __host__ void Allocate(size_t width, size_t height)
		{
			m_Width = width;
			m_Height = height;
			m_Size = width * height;

			cudaError_t err = cudaMallocManaged(&m_Data, m_Size * sizeof(Type));

			if (err != cudaError::cudaSuccess)
			{
				printf("Error while allocating memory on the device with size of %llu, with error code: 0x%016X\n", m_Size * sizeof(Type), err);
			}

			m_NumBlocks = dim3(m_Width / m_ThreadsPerBlock.x, m_Height / m_ThreadsPerBlock.y);
		}

		inline void SetBusyFlag(bool busy) { m_Flags = (m_Flags & ~FLAG_IS_BUSY) | ((busy ? 1 : 0) << 1); }

	private:

		uint8_t m_Flags = 0x0;
		dim3 m_ThreadsPerBlock, m_NumBlocks;
	};

}
#endif