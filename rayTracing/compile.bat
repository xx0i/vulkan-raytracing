C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 rayGen.rgen -o rgen.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 miss.rmiss -o rmiss.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 closesthit.rchit -o rchit.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V -I. --target-env vulkan1.2 quadClosesthit.rchit -o quadrchit.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V -I. --target-env vulkan1.2 sphereClosesthit.rchit -o sphererchit.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 anyhit.rahit -o rahit.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 denoiserAndColouring.comp -o comp.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 intersection.rint -o rint.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 quadIntersection.rint -o quad.spv
C:/VulkanSDK/1.4.309.0/Bin/glslangValidator.exe -V --target-env vulkan1.2 sphereIntersection.rint -o sphere.spv
pause