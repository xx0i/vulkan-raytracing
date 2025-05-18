# My Vulkan raytracing project!!!
I have created a vulkan project from scratch building up from window creation to rasterization, to raytracing (Still in progress!)

## To run the project:
- please have vulkan installed on your machine.
- once the project is cloned go to the properties and please add the following dependencies under the C/C++ → General → Additional Include Directories:
  
      $(SolutionDir)..\libraries\tiny_obj_loader
      $(SolutionDir)..\libraries\stb_image
      C:\VulkanSDK\1.4.309.0\Include  (or wherever your version of vulkan is installed)
      $(SolutionDir)..\libraries\glm-1.0.1\glm
      $(SolutionDir)..\libraries\glfw-3.4.bin.WIN64\include
- in the  Linker → General → Additional Library Directories please add the following
  
      C:\VulkanSDK\1.4.309.0\Lib
      $(SolutionDir)..\libraries\glfw-3.4.bin.WIN64\lib-vc2022
- after applying these changes you should be able to run the project!
