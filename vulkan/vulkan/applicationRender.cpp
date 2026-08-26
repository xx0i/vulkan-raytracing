#include "Application.h"

// ============================================================
// entry point / lifecycle
// ============================================================

void application::run()
{
	windowInitalization();
	vulkanInitalization();
	mainLoop();
	cleanup();
}

void application::windowInitalization()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	window = glfwCreateWindow(width, height, "vulkan", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, frameBufferResizeCallback);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void application::vulkanInitalization()
{
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();
	createTimestampQueryPools();
	createSwapChain();
	createImageViews();
	createRenderPass();
	createImGuiRenderPass();
	updateCameraVectors();
	//createDescriptorSetLayout();
	createRayTracingDescriptorSetLayout();
	createAlphaDescriptorSetLayout();
	createComputeDescriptorSetLayout();
	//createGraphicsPipeline();
	createRayTracingPipeline();
	createComputePipeline();
	createCommandPool();
	createColourResources();
	createDepthResources();
	createFrameBuffers();
	createImguiFrameBuffers();
	createTextureImage();
	createTextureImageView();
	createTextureSampler();
	createStoreImage();
	createStoreImageView();
	createAlphaImage();
	createAlphaImageView();
	createAlphaSampler();
	createHistorySampler();
	createComputeImageA();
	createComputeImageViewA();
	createComputeImageB();
	createComputeImageViewB();
	createAccumulationImageA();
	createAccumulationImageViewA();
	createAccumulationImageB();
	createAccumulationImageViewB();
	createNormalImage();
	createNormalImageView();
	createAlbedoImage();
	createAlbedoImageView();
	createVelocityImage();
	createVelocityImageView();
	createMomentImageA();
	createMomentImageViewA();
	createMomentImageB();
	createMomentImageViewB();
	createPrevNormalImage();
	createPrevNormalImageView();
	loadModel();
	//simpleDraw();
	drawShapes();
	createVertexBuffer();
	createIndexBuffer();
	createAABBBuffer();
	if (spheres.size() > 0)
	{
		createSphereBuffer();
	}
	createMaterialBuffer();
	if (quads.size() > 0)
	{
		createQuadBuffer();
	}
	createGeoTypeBuffer();
	createAabbObjectsBuffer();
	createAccerlerationStructures();
	createUniformBuffer();
	createShaderBindingTables();
	//createDescriptorPool();
	createRayTracingDescriptorPool();
	createComputeDescriptorPool();
	createImguiDescriptorPool();
	imguiInitialization();
	//createDescriptorSets();
	createRayTracingDescriptorSets();
	createAlphaDescriptorSets();
	createComputeDescriptorSets();
	createCommandBuffers();
	createSyncObjects();
}

void application::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		{
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}

		glfwPollEvents();
		drawFrameRayTracing();
	}

	vkDeviceWaitIdle(device);
}

void application::cleanupSwapChain()
{
	for (auto framebuffer : swapChainFrameBuffers)
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	swapChainFrameBuffers.clear();

	for (auto framebuffer : imguiFrameBuffers)
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	imguiFrameBuffers.clear();

	if (imguiRenderPass != VK_NULL_HANDLE)
	{
		vkDestroyRenderPass(device, imguiRenderPass, nullptr);
		imguiRenderPass = VK_NULL_HANDLE;
	}

	if (rayTracingAndAlphaDescriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, rayTracingAndAlphaDescriptorPool, nullptr);
		rayTracingAndAlphaDescriptorPool = VK_NULL_HANDLE;
	}

	if (computeDescriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, computeDescriptorPool, nullptr);
		computeDescriptorPool = VK_NULL_HANDLE;
	}

	vkDestroySampler(device, alphaSampler, nullptr);

	vkDestroyImageView(device, storeImageView, nullptr);
	vkDestroyImage(device, storeImage, nullptr);
	vkFreeMemory(device, storeImageMemory, nullptr);

	vkDestroyImageView(device, alphaImageView, nullptr);
	vkDestroyImage(device, alphaImage, nullptr);
	vkFreeMemory(device, alphaImageMemory, nullptr);

	vkDestroyImageView(device, computeImageViewA, nullptr);
	vkDestroyImage(device, computeImageA, nullptr);
	vkFreeMemory(device, computeImageMemoryA, nullptr);

	vkDestroyImageView(device, computeImageViewB, nullptr);
	vkDestroyImage(device, computeImageB, nullptr);
	vkFreeMemory(device, computeImageMemoryB, nullptr);

	vkDestroyImageView(device, accumulationImageViewA, nullptr);
	vkDestroyImage(device, accumulationImageA, nullptr);
	vkFreeMemory(device, accumulationImageMemoryA, nullptr);

	vkDestroyImageView(device, accumulationImageViewB, nullptr);
	vkDestroyImage(device, accumulationImageB, nullptr);
	vkFreeMemory(device, accumulationImageMemoryB, nullptr);

	vkDestroyImageView(device, depthImageView, nullptr);
	vkDestroyImage(device, depthImage, nullptr);
	vkFreeMemory(device, depthImageMemory, nullptr);

	vkDestroyImageView(device, colourImageView, nullptr);
	vkDestroyImage(device, colourImage, nullptr);
	vkFreeMemory(device, colourImageMemory, nullptr);

	vkDestroyImageView(device, normalImageView, nullptr);
	vkDestroyImage(device, normalImage, nullptr);
	vkFreeMemory(device, normalImageMemory, nullptr);

	vkDestroyImageView(device, albedoImageView, nullptr);
	vkDestroyImage(device, albedoImage, nullptr);
	vkFreeMemory(device, albedoImageMemory, nullptr);

	vkDestroyImageView(device, velocityImageView, nullptr);
	vkDestroyImage(device, velocityImage, nullptr);
	vkFreeMemory(device, velocityImageMemory, nullptr);

	vkDestroyImageView(device, momentImageViewA, nullptr);
	vkDestroyImage(device, momentImageA, nullptr);
	vkFreeMemory(device, momentImageAMemory, nullptr);

	vkDestroyImageView(device, momentImageViewB, nullptr);
	vkDestroyImage(device, momentImageB, nullptr);
	vkFreeMemory(device, momentImageBMemory, nullptr);

	vkDestroyImageView(device, prevNormalImageView, nullptr);
	vkDestroyImage(device, prevNormalImage, nullptr);
	vkFreeMemory(device, prevNormalImageMemory, nullptr);

	for (auto imageView : swapChainImageViews)
		vkDestroyImageView(device, imageView, nullptr);
	swapChainImageViews.clear();

	if (swapChain != VK_NULL_HANDLE)
	{
		vkDestroySwapchainKHR(device, swapChain, nullptr);
		swapChain = VK_NULL_HANDLE;
	}
}

void application::cleanup()
{
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);

	cleanupSwapChain();

	vkDestroyPipeline(device, graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	vkDestroyRenderPass(device, renderPass, nullptr);

	vkDestroyPipeline(device, rayTracingPipeline, nullptr);
	vkDestroyPipelineLayout(device, rayTracingPipelineLayout, nullptr);

	vkDestroyPipeline(device, computePipeline, nullptr);
	vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		vkDestroyBuffer(device, uniformBuffers[i], nullptr);
		vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
	}

	vkDestroyDescriptorPool(device, descriptorPool, nullptr);

	vkDestroySampler(device, textureSampler, nullptr);
	vkDestroyImageView(device, textureImageView, nullptr);
	vkDestroyImage(device, textureImage, nullptr);
	vkFreeMemory(device, textureImageMemory, nullptr);

	vkDestroySampler(device, historySampler, nullptr);

	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

	vkDestroyDescriptorSetLayout(device, rayTracingDescriptorSetLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, alphaDescriptorSetLayout, nullptr);

	vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);

	vkDestroyBuffer(device, vertexBuffer, nullptr);
	vkFreeMemory(device, vertexBufferMemory, nullptr);

	vkDestroyBuffer(device, indexBuffer, nullptr);
	vkFreeMemory(device, indexBufferMemory, nullptr);

	vkDestroyBuffer(device, aabbBuffer, nullptr);
	vkFreeMemory(device, aabbBufferMemory, nullptr);

	vkDestroyBuffer(device, sphereBuffer, nullptr);
	vkFreeMemory(device, sphereBufferMemory, nullptr);

	vkDestroyBuffer(device, materialBuffer, nullptr);
	vkFreeMemory(device, materialBufferMemory, nullptr);

	vkDestroyBuffer(device, quadBuffer, nullptr);
	vkFreeMemory(device, quadBufferMemory, nullptr);

	vkDestroyBuffer(device, geoTypeBuffer, nullptr);
	vkFreeMemory(device, geoTypeBufferMemory, nullptr);

	vkDestroyBuffer(device, aabbObjectsBuffer, nullptr);
	vkFreeMemory(device, aabbObjectsBufferMemory, nullptr);

	vkDestroyBuffer(device, blasBuffer, nullptr);
	vkFreeMemory(device, blasMemory, nullptr);
	DestroyAccelerationStructureKHR(device, blas, nullptr);

	for (auto& obj : aabbObjects)
	{
		if (obj.blas != VK_NULL_HANDLE)
		{
			DestroyAccelerationStructureKHR(device, obj.blas, nullptr);
		}
		if (obj.blasBuffer != VK_NULL_HANDLE)
		{
			vkDestroyBuffer(device, obj.blasBuffer, nullptr);
		}
		if (obj.blasMemory != VK_NULL_HANDLE)
		{
			vkFreeMemory(device, obj.blasMemory, nullptr);
		}
	}

	vkDestroyBuffer(device, tlasBuffer, nullptr);
	vkFreeMemory(device, tlasMemory, nullptr);
	DestroyAccelerationStructureKHR(device, tlas, nullptr);

	vkDestroyBuffer(device, shaderBindingTableBuffer, nullptr);
	vkFreeMemory(device, shaderBindingTableBufferMemory, nullptr);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(device, inFlightFences[i], nullptr);
	}

	for (auto& semaphore : renderFinishedSemaphores) {
		vkDestroySemaphore(device, semaphore, nullptr);
	}

	vkDestroyCommandPool(device, commandPool, nullptr);

	for (auto pool : timeStampQueryPools) {
		vkDestroyQueryPool(device, pool, nullptr);
	}

	vkDestroyDevice(device, nullptr);

	if (enableValidationLayers)
	{
		DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	}

	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
}

void application::recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(device);

	cleanupSwapChain();

	createSwapChain();
	createImageViews();
	createColourResources();
	createDepthResources();

	createStoreImage();
	createStoreImageView();

	createAlphaImage();
	createAlphaImageView();
	createAlphaSampler();

	createComputeImageA();
	createComputeImageViewA();

	createComputeImageB();
	createComputeImageViewB();

	createAccumulationImageA();
	createAccumulationImageViewA();

	createAccumulationImageB();
	createAccumulationImageViewB();

	createNormalImage();
	createNormalImageView();

	createAlbedoImage();
	createAlbedoImageView();

	createVelocityImage();
	createVelocityImageView();

	createMomentImageA();
	createMomentImageViewA();

	createMomentImageB();
	createMomentImageViewB();

	createPrevNormalImage();
	createPrevNormalImageView();

	VkClearColorValue clearColor = { { 0.0f, 0.0f, 0.0f, 0.0f } };

	VkCommandBuffer cmd = beginSingleTimeCommands();

	VkImageSubresourceRange range{};
	range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	range.baseMipLevel = 0;
	range.levelCount = 1;
	range.baseArrayLayer = 0;
	range.layerCount = 1;

	vkCmdClearColorImage(
		cmd,
		accumulationImageA,
		VK_IMAGE_LAYOUT_GENERAL, // or transition from UNDEFINED → GENERAL first
		&clearColor,
		1,
		&range);

	vkCmdClearColorImage(
		cmd,
		accumulationImageB,
		VK_IMAGE_LAYOUT_GENERAL, // or transition from UNDEFINED → GENERAL first
		&clearColor,
		1,
		&range);

	endSingleTimeCommands(cmd);

	frameCounter = 0;

	createFrameBuffers();

	createRayTracingDescriptorPool();
	createComputeDescriptorPool();

	createRayTracingDescriptorSets();
	createAlphaDescriptorSets();
	createComputeDescriptorSets();

	createImGuiRenderPass();
	createImguiFrameBuffers();

	ImGui_ImplVulkan_SetMinImageCount(static_cast<uint32_t>(swapChainImages.size()));
}

// ============================================================
// per-frame update / draw
// ============================================================

void application::updateUniformBuffer(uint32_t currentFrame, bool cameraMoved)
{
	static std::vector<glm::mat4> prevViews(maxFramesInFlight, glm::mat4(1.0f));
	static std::vector<glm::mat4> prevProjs(maxFramesInFlight, glm::mat4(1.0f));
	static bool historyInitialized = false;

	uniformBufferObject ubo{};
	ubo.model = glm::mat4(1.0f);

	ubo.view = glm::lookAt(camera.position, camera.position + camera.front, camera.up);

	glm::mat4 cleanProj = glm::perspective(
		glm::radians(60.0f),
		swapChainExtent.width / static_cast<float>(swapChainExtent.height),
		0.1f, 512.0f
	);
	cleanProj[1][1] *= -1.0f;

	ubo.projUnjittered = cleanProj;
	ubo.proj = cleanProj;

	if (!historyInitialized)
	{
		for (uint32_t i = 0; i < maxFramesInFlight; ++i)
		{
			prevViews[i] = ubo.view;
			prevProjs[i] = ubo.projUnjittered;
		}
		historyInitialized = true;
	}

	ubo.prevView = prevViews[currentFrame];
	ubo.prevProjUnjittered = prevProjs[currentFrame];

	memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));

	prevViews[currentFrame] = ubo.view;
	prevProjs[currentFrame] = ubo.projUnjittered;
}

void application::drawFrame()
{

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Settings");
	ImGui::Text("Frame: %d", currentFrame);
	ImGui::End();

	ImGui::Render();

	vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		throw std::runtime_error("failed to acquire swap chain image");
	}

	vkResetFences(device, 1, &inFlightFences[currentFrame]);

	vkResetCommandBuffer(commandBuffers[currentFrame], 0);
	recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = nullptr;

	result = vkQueuePresentKHR(presentQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frameBufferResized)
	{
		recreateSwapChain();
		frameBufferResized = false;
	}
	else if (result != VK_SUCCESS)
	{
		throw std::runtime_error("failed to present swap chain image");
	}

	currentFrame = (currentFrame + 1) % maxFramesInFlight;
}

void application::drawFrameRayTracing()
{
	vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
	if (frameCounter > maxFramesInFlight)
	{
		vkGetQueryPoolResults(
			device,
			timeStampQueryPools[currentFrame],
			0,
			timeStamps.size(),
			timeStamps.size() * sizeof(uint64_t),
			timeStamps.data(),
			sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	}

	vkResetQueryPool(device, timeStampQueryPools[currentFrame], 0, timeStamps.size());

	auto calculateTime = [&](size_t startIdx, size_t endIdx) {
		if (timeStamps[endIdx] > timeStamps[startIdx]) {
			return (timeStamps[endIdx] - timeStamps[startIdx]) * timestampPeriod / 1000000.0;
		}
		return 0.0;
		};

	double rtTime = calculateTime(0, 1);
	double computeTime = calculateTime(2, 3);
	double imguiTime = calculateTime(4, 5);

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
		ImGuiWindowFlags_AlwaysAutoResize |
		ImGuiWindowFlags_NoSavedSettings |
		ImGuiWindowFlags_NoFocusOnAppearing |
		ImGuiWindowFlags_NoNav |
		ImGuiWindowFlags_NoMove;

	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.3f);
	ImGui::Begin("Info", nullptr, window_flags);
	ImGui::Text("Frame: %d", currentFrame);
	ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
	ImGui::Text("Frame Time: %.3f ms", 1000.0f / ImGui::GetIO().Framerate);
	ImGui::End();

	float screenHeight = ImGui::GetIO().DisplaySize.y;
	ImGui::SetNextWindowPos(ImVec2(10, screenHeight - 10), ImGuiCond_Always, ImVec2(0, 1));
	ImGui::SetNextWindowBgAlpha(0.3f);
	ImGui::Begin("Performance", nullptr, window_flags);
	ImGui::Text("Ray Tracing: %.3f ms", rtTime);
	ImGui::Text("Compute: %.3f ms", computeTime);
	ImGui::Text("ImGui Render: %.3f ms", imguiTime);
	ImGui::Separator();
	ImGui::Text("Total GPU:   %.3f ms", rtTime + computeTime + imguiTime);

	// --- RESOLUTION SELECTOR ---
	ImGui::Separator();
	const char* resolutions[] = {
		"1280x720 (720p)",
		"1920x1080 (1080p)",
		"2560x1440 (1440p)",
		"3840x2160 (4K)"
	};
	static int currentResIndex = 1; // Default: 1080p

	if (ImGui::Combo("Resolution", &currentResIndex, resolutions, IM_ARRAYSIZE(resolutions)))
	{
		int targetWidth = 1920;
		int targetHeight = 1080;

		switch (currentResIndex) {
		case 0: targetWidth = 1280; targetHeight = 720;  break;
		case 1: targetWidth = 1920; targetHeight = 1080; break;
		case 2: targetWidth = 2560; targetHeight = 1440; break;
		case 3: targetWidth = 3840; targetHeight = 2160; break;
		}

		glfwSetWindowSize(window, targetWidth, targetHeight);
	}

	ImGui::End();

	ImGui::Render();

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frameBufferResized)
	{
		frameBufferResized = false;
		recreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		throw std::runtime_error("failed to acquire swap chain image");
	}

	auto currentTime = std::chrono::high_resolution_clock::now();
	float deltaTime = std::chrono::duration<float>(currentTime - startTime).count();
	startTime = currentTime;

	forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
	backward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
	left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
	right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
	up = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
	down = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;

	bool keyboardMoved = processKeyboard(deltaTime);

	bool mouseMoved = false;
	if (std::abs(accumulatedDeltaX) > 1e-4f || std::abs(accumulatedDeltaY) > 1e-4f)
	{
		processMouse(accumulatedDeltaX, accumulatedDeltaY);
		accumulatedDeltaX = 0.0f;
		accumulatedDeltaY = 0.0f;
		mouseMoved = true;
	}

	bool cameraMoved = keyboardMoved || mouseMoved;

	frameCounter++;
	totalFrameCount++;

	updateUniformBuffer(currentFrame, cameraMoved);

	vkResetFences(device, 1, &inFlightFences[currentFrame]);

	vkResetCommandBuffer(commandBuffers[currentFrame], 0);

	recordRayTracingCommandBuffer(commandBuffers[currentFrame], imageIndex, frameCounter, totalFrameCount);

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer");
	}

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = nullptr;

	result = vkQueuePresentKHR(presentQueue, &presentInfo);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frameBufferResized)
	{
		recreateSwapChain();
		frameBufferResized = false;
	}
	else if (result != VK_SUCCESS)
	{
		throw std::runtime_error("failed to present swap chain image");
	}

	currentFrame = (currentFrame + 1) % maxFramesInFlight;
}