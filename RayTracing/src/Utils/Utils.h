#pragma once

#if 1
	#define __UTILS_HAS_FASTCALL
	#define __UTILS_HAS_INLINE
	#define __UTILS_FORCEINLINE
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

namespace Utils {

}