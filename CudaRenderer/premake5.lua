project "CudaRenderer"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   local cuda = {}
   cuda["path"] = os.getenv("CUDA_PATH")
   if cuda["path"] == nil then 
      print("Couldn't find cuda in your machine, install it and then try again\nUse this link: https://developer.nvidia.com/cuda-downloads")
      os.exit()
   end

   cuda["version"] = cuda["path"]:match("\\v([%d%.]+)$")
   
   print("Detected cuda version: " .. cuda["version"])

   ignoredefaultlibraries { "LIBCMT" }

   buildcustomizations ("BuildCustomizations/CUDA " .. cuda["version"])
   files { "src/**.h", "src/**.cpp", "src/**.cuh" }
   cudaFiles { "src/**.cu" }
   removefiles { "src/Utils/**" }
   externalwarnings "Off"
   cudaCompilerOptions { "-arch=all", "-t0" }
   
   cudaIntDir ("../bin-int/" .. outputdir .. "/cuda/%%(Filename)%%(Extension).obj")

   includedirs
   {
      "../Walnut/vendor/imgui",
      "../Walnut/vendor/glfw/include",
      "../Walnut/vendor/glm",
      "../Walnut/vendor/stb_image",
      "../Walnut/Walnut/src",

      "../SGOL/SGOL/include",

      "%{IncludeDir.VulkanSDK}",
   }

   links
   {
       "Walnut",
       "cudart_static.lib",
   }

   targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

   filter "system:windows"
      systemversion "latest"
      defines { "WL_PLATFORM_WINDOWS", "_CRT_SECURE_NO_WARNINGS" }

   filter "configurations:Debug"
      defines { "WL_DEBUG", "_CRT_SECURE_NO_WARNINGS", "_DEBUG"  }
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      defines { "WL_RELEASE", "_CRT_SECURE_NO_WARNINGS", "NDEBUG"  }
      runtime "Release"
      optimize "Speed"
      symbols "On"
      cudaFastMath "On"
      cudaGenLineInfo "On"

   filter "configurations:Dist"
      kind "WindowedApp"
      defines { "WL_DIST", "_CRT_SECURE_NO_WARNINGS", "NDEBUG"  }
      runtime "Release"
      optimize "Speed"
      symbols "Off"
      cudaFastMath "On"