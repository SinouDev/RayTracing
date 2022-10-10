#pragma once

#define __UTILS_ENABLE_ALL

#ifdef __UTILS_ENABLE_ALL
	#define __UTILS_HAS_FASTCALL
	#define __UTILS_HAS_INLINE
	#define __UTILS_FORCEINLINE
	#define __UTILS_HAS_NODISCARD
#endif

#ifdef __UTILS_HAS_FASTCALL
	#define __UTILS_FASTCALL __fastcall
#else
	#define __UTILS_FASTCALL
#endif //__UTILS_HAS_FASTCALL

#ifdef __UTILS_HAS_INLINE
	#ifdef __UTILS_FORCEINLINE
		#define __UTILS_INLINE __forceinline
	#else
		#define __UTILS_INLINE inline
	#endif
#else
	#define __UTILS_INLINE
#endif //__UTILS_HAS_INLINE

#ifdef __UTILS_HAS_NODISCARD
	#define __UTILS_NODISCARD [[nodiscard]]
#else
	#define __UTILS_NODISCARD
#endif

#ifdef __UTILS_ENABLE_ALL
	#undef __UTILS_HAS_FASTCALL
	#undef __UTILS_HAS_INLINE
	#undef __UTILS_FORCEINLINE
	#undef __UTILS_HAS_NODISCARD
#endif

namespace Utils {

}