project "RayTracing"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   targetdir "bin/%{cfg.buildcfg}"
   staticruntime "off"

   files { "src/**.h", "src/**.cpp" }

   

   includedirs
   {
      "../Walnut/vendor/imgui",
      "../Walnut/vendor/glfw/include",
      "../Walnut/vendor/glm",
      "../Walnut/vendor/stb_image",
      "../Walnut/vendor/benchmark/include",
      "../Walnut/Walnut/src",
      "src",

      "%{IncludeDir.VulkanSDK}",
   }

   links
   {
       "Walnut"
   }

   targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
   objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

   filter "system:windows"
      systemversion "latest"
      defines { "WL_PLATFORM_WINDOWS", "_CRT_SECURE_NO_WARNINGS" }

   filter "configurations:Debug"
      defines { "WL_DEBUG", "_CRT_SECURE_NO_WARNINGS"  }
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      defines { "WL_RELEASE", "_CRT_SECURE_NO_WARNINGS"  }
      runtime "Release"
      optimize "On"
      symbols "On"

   filter "configurations:Dist"
      kind "WindowedApp"
      defines { "WL_DIST", "_CRT_SECURE_NO_WARNINGS"  }
      runtime "Release"
      optimize "On"
      symbols "Off"