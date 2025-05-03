-- premake5.lua

flags {
   "RelativeLinks",
   "MultiProcessorCompile"
 }

workspace "RayTracing"
   architecture "x64"
   configurations { "Debug", "Release", "Dist" }
   startproject "CudaRenderer"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
include "Walnut/WalnutExternal.lua"

require "./premake5-cuda/premake5-cuda"
include "CudaRenderer"


group "Dependencies"
group ""