#pragma once
#include "Common.h"
#include "Types.h"
#include "VulkanExtensions.h"

class application
{
public:
	void run();

private:

	//variables
	GLFWwindow* window;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
	VkDevice device;

	VkQueue graphicsQueue;
	VkQueue presentQueue;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFrameBuffers;

	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	VkDescriptorSetLayout rayTracingDescriptorSetLayout;
	VkDescriptorSetLayout alphaDescriptorSetLayout;
	VkDescriptorSetLayout computeDescriptorSetLayout;

	VkPipelineLayout rayTracingPipelineLayout;
	VkPipeline rayTracingPipeline;

	VkPipelineLayout computePipelineLayout;
	VkPipeline computePipeline;

	VkCommandPool commandPool;

	VkImage colourImage;
	VkDeviceMemory colourImageMemory;
	VkImageView colourImageView;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	uint32_t mipLevels;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	VkImage storeImage;
	VkDeviceMemory storeImageMemory;
	VkImageView storeImageView;

	VkImage alphaImage;
	VkDeviceMemory alphaImageMemory;
	VkImageView alphaImageView;
	VkSampler alphaSampler;

	VkSampler historySampler;

	VkImage computeImageA;
	VkDeviceMemory computeImageMemoryA;
	VkImageView computeImageViewA;

	VkImage computeImageB;
	VkDeviceMemory computeImageMemoryB;
	VkImageView computeImageViewB;

	VkImage accumulationImageA;
	VkDeviceMemory accumulationImageMemoryA;
	VkImageView accumulationImageViewA;

	VkImage accumulationImageB;
	VkDeviceMemory accumulationImageMemoryB;
	VkImageView accumulationImageViewB;

	VkImage normalImage;
	VkDeviceMemory normalImageMemory;
	VkImageView normalImageView;

	VkImage albedoImage;
	VkDeviceMemory albedoImageMemory;
	VkImageView albedoImageView;

	VkImage velocityImage;
	VkDeviceMemory velocityImageMemory;
	VkImageView velocityImageView;

	VkImage prevNormalImage;
	VkDeviceMemory prevNormalImageMemory;
	VkImageView prevNormalImageView;

	VkImage momentImageA;
	VkDeviceMemory momentImageAMemory;
	VkImageView momentImageViewA;

	VkImage momentImageB;
	VkDeviceMemory momentImageBMemory;
	VkImageView momentImageViewB;

	VkImage prevNormalDepthImage;
	VkDeviceMemory prevNormalDepthImageMemory;
	VkImageView prevNormalDepthImageView;

	std::vector<vertex> vertices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkDeviceAddress vertexAddress;

	std::vector<uint32_t> indices;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	VkDeviceAddress indexAddress;

	VkBuffer shaderBindingTableBuffer;
	VkDeviceMemory shaderBindingTableBufferMemory;
	VkDeviceAddress shaderBindingTableAddress;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	VkDescriptorPool rayTracingAndAlphaDescriptorPool;
	std::vector<VkDescriptorSet> rayTracingDescriptorSets;
	std::vector<VkDescriptorSet> alphaDescriptorSets;

	VkDescriptorPool computeDescriptorPool;
	std::vector<VkDescriptorSet> computeDescriptorSets;

	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0;

	VkBuffer blasBuffer;
	VkDeviceMemory blasMemory;
	VkAccelerationStructureKHR blas;

	VkBuffer tlasBuffer;
	VkDeviceMemory tlasMemory;
	VkAccelerationStructureKHR tlas;

	VkStridedDeviceAddressRegionKHR raygenRegion{};
	VkStridedDeviceAddressRegionKHR missRegion{};
	VkStridedDeviceAddressRegionKHR hitRegion{};
	VkStridedDeviceAddressRegionKHR callableRegion{};

	std::vector<VkAabbPositionsKHR> aabbs;
	VkBuffer aabbBuffer;
	VkDeviceMemory aabbBufferMemory;
	VkDeviceAddress aabbAddress;

	std::vector<sphere> spheres;
	VkBuffer sphereBuffer;
	VkDeviceMemory sphereBufferMemory;
	VkDeviceAddress sphereAddress;

	std::vector<material> materials;
	VkBuffer materialBuffer;
	VkDeviceMemory materialBufferMemory;
	VkDeviceAddress materialAddress;

	std::vector<quad> quads;
	VkBuffer quadBuffer;
	VkDeviceMemory quadBufferMemory;
	VkDeviceAddress quadAddress;

	std::vector<geometryType> geoTypes;
	VkBuffer geoTypeBuffer;
	VkDeviceMemory geoTypeBufferMemory;
	VkDeviceAddress geoTypeAddress;

	std::vector<aabbObject> aabbObjects;
	VkBuffer aabbObjectsBuffer;
	VkDeviceMemory aabbObjectsBufferMemory;
	VkDeviceAddress aabbObjectsAddress;

	std::vector<AabbObjectGPU> gpuAabbs;

	VkDescriptorPool imguiDescriptorPool;
	VkRenderPass imguiRenderPass;
	std::vector<VkFramebuffer> imguiFrameBuffers;

	VkExtent3D extent;

	std::vector<VkQueryPool> timeStampQueryPools;
	std::vector<uint64_t> timeStamps{};

	camera camera
	{
		glm::vec3(4.46082, 3.29564, 1.53025),     // position
		glm::radians(-145.077f),				  // yaw
		glm::radians(-14.051f),					  // pitch
		2.0f,									  // speed
		0.001f,									  // sensitivity
		glm::vec3(0.0f),						  // front (will be set by updateCameraVectors)
		glm::vec3(0.0f, 0.0f, 1.0f),			  // up
		glm::vec3(0.0f),						  // right
	};

	glm::mat4 globalPrevView{ 1.0f };
	glm::mat4 globalPrevProj{ 1.0f };
	glm::mat4 globalPrevProjUnjittered{ 1.0f };
	bool historyInitialized = false;

	float timestampPeriod;
	double lastMouseX = 0.0;
	double lastMouseY = 0.0;
	bool firstMouse = true;
	bool forward;
	bool backward;
	bool left;
	bool right;
	bool up;
	bool down;
	bool mouseMoved;
	bool isMouseCaptured = true;
	float accumulatedDeltaX = 0.0f;
	float accumulatedDeltaY = 0.0f;

	bool frameBufferResized = false;

	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

	uint32_t frameCounter;
	uint32_t missShaderColouring;
	uint32_t currentStepSize;
	uint32_t totalFrameCount = 0;

	std::vector<glm::mat4> prevViewMatrices;
	std::vector<glm::mat4> prevProjMatrices;

	// --- windowing / callbacks ---
	void windowInitalization();
	static void frameBufferResizeCallback(GLFWwindow* window, int width, int height);
	static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
	void handleMouseCallback(double xpos, double ypos);

	// --- top-level init / loop / cleanup ---
	void vulkanInitalization();
	void mainLoop();
	void cleanupSwapChain();
	void cleanup();
	void recreateSwapChain();

	// --- instance / debug / device ---
	void createInstance();
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
	void setupDebugMessenger();
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createTimestampQueryPools();

	// --- swapchain ---
	void createSwapChain();
	void createImageViews();
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	swapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

	// --- render passes / framebuffers ---
	void createRenderPass();
	void createImGuiRenderPass();
	void createFrameBuffers();
	void createImguiFrameBuffers();

	// --- descriptor set layouts ---
	void createDescriptorSetLayout();
	void createRayTracingDescriptorSetLayout();
	void createAlphaDescriptorSetLayout();
	void createComputeDescriptorSetLayout();

	// --- pipelines ---
	void createGraphicsPipeline();
	void createRayTracingPipeline();
	void createComputePipeline();
	void createShaderBindingTables();

	// --- image / resource creation helpers ---
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat findDepthFormat();
	bool hasStencilComponenet(VkFormat format);

	void createColourResources();
	void createDepthResources();

	void createTextureImage();
	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
	VkSampleCountFlagBits getMaxUsableSampleCount();

	void createStoreImage();
	void createAlphaImage();
	void createComputeImageA();
	void createComputeImageB();
	void createAccumulationImageA();
	void createAccumulationImageB();
	void createNormalImage();
	void createAlbedoImage();
	void createVelocityImage();
	void createMomentImageA();
	void createMomentImageB();
	void createPrevNormalImage();

	void createTextureImageView();
	void createStoreImageView();
	void createAlphaImageView();
	void createComputeImageViewA();
	void createComputeImageViewB();
	void createAccumulationImageViewA();
	void createAccumulationImageViewB();
	void createNormalImageView();
	void createAlbedoImageView();
	void createVelocityImageView();
	void createPrevNormalImageView();
	void createMomentImageViewA();
	void createMomentImageViewB();

	void createTextureSampler();
	void createAlphaSampler();
	void createHistorySampler();

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags imageAspectFlags, uint32_t mipLevels);
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSample, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usageFlags,
		VkMemoryPropertyFlags propertyFlags, VkImage& image, VkDeviceMemory& imageMemory);

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
	void transitionImageLayoutInCommandBuffer(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subresourceRange);
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	// --- scene construction ---
	void simpleDraw();
	void drawShapes();
	void makeBox(glm::vec3 p0, glm::vec3 p1);
	void makeRotatedBox(const glm::vec3& pMin, const glm::vec3& pMax, float angle);
	inline float random_float(float min = 0.0f, float max = 1.0f);
	inline glm::vec3 random_vec3(float min = 0.0f, float max = 1.0f);
	void loadModel();

	// --- buffers ---
	void createVertexBuffer();
	void createIndexBuffer();
	void createAABBBuffer();
	void createSphereBuffer();
	void createMaterialBuffer();
	void createQuadBuffer();
	void createGeoTypeBuffer();
	void createAabbObjectsBuffer();

	// --- acceleration structures ---
	void createAccerlerationStructures();
	void createBLAStriangle();
	void createBLASaabb();
	VkAccelerationStructureKHR createBLASForAABB(const VkAabbPositionsKHR& aabb, VkDeviceAddress& deviceAddr, VkBuffer& buffer, VkDeviceMemory& memory);
	void createTLAS();
	void createMultiInstanceTLAS();

	// --- uniform buffer / descriptor pools & sets ---
	void createUniformBuffer();
	void createDescriptorPool();
	void createRayTracingDescriptorPool();
	void createComputeDescriptorPool();
	void createImguiDescriptorPool();
	void imguiInitialization();
	void createDescriptorSets();
	void createRayTracingDescriptorSets();
	void createAlphaDescriptorSets();
	void createComputeDescriptorSets();

	// --- generic buffer helpers ---
	void createBuffer(VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize deviceSize);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	VkDeviceAddress findBufferDeviceAddress(VkDevice device, VkBuffer buffer);
	VkDeviceAddress findAccelerationStructureDeviceAddress(VkDevice device, VkAccelerationStructureKHR accelerationStructure);

	// --- command buffers / recording ---
	void createCommandPool();
	void createCommandBuffers();
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
	void recordRayTracingCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t progressiveFrameCount, uint32_t globalFrameCount);

	// --- sync objects ---
	void createSyncObjects();

	// --- camera / input ---
	void updateCameraVectors();
	void processMouse(float deltaX, float deltaY);
	bool processKeyboard(float deltaTime);

	// --- per-frame update / draw ---
	void updateUniformBuffer(uint32_t currentFrame, bool cameraMoved);
	void drawFrame();
	void drawFrameRayTracing();

	// --- shader modules ---
	VkShaderModule createShaderModule(const std::vector<char>& code);

	// --- device suitability / queue families / extensions ---
	bool isDeviceSuitable(VkPhysicalDevice device);
	bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	queueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
	std::vector<const char*> getRequiredExtensions();
	bool checkValidationLayerSupport();

	// --- misc utility ---
	static std::vector<char> readFile(const std::string& fileName);
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
};
