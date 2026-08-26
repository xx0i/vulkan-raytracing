#include "Application.h"

// ============================================================
// uniform buffer
// ============================================================

void application::createUniformBuffer()
{
	VkDeviceSize bufferSize = sizeof(uniformBufferObject);

	uniformBuffers.resize(maxFramesInFlight);
	uniformBuffersMemory.resize(maxFramesInFlight);
	uniformBuffersMapped.resize(maxFramesInFlight);
	prevViewMatrices.resize(maxFramesInFlight);
	prevProjMatrices.resize(maxFramesInFlight);

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

		vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
	}
}

// ============================================================
// descriptor pools
// ============================================================

void application::createDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 2> descriptorPoolSizes{};
	descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorPoolSizes[0].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorPoolSizes[1].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	VkDescriptorPoolCreateInfo descriptorPoolInfo{};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
	descriptorPoolInfo.pPoolSizes = descriptorPoolSizes.data();
	descriptorPoolInfo.maxSets = static_cast<uint32_t>(maxFramesInFlight);

	if (vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor pool");
	}
}

void application::createRayTracingDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 7> descriptorPoolSizes{};
	descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	descriptorPoolSizes[0].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	descriptorPoolSizes[1].descriptorCount = static_cast<uint32_t>(maxFramesInFlight * 4);

	descriptorPoolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorPoolSizes[2].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	descriptorPoolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorPoolSizes[3].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	descriptorPoolSizes[4].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	descriptorPoolSizes[4].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	descriptorPoolSizes[5].type = VK_DESCRIPTOR_TYPE_SAMPLER;
	descriptorPoolSizes[5].descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

	descriptorPoolSizes[6].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorPoolSizes[6].descriptorCount = static_cast<uint32_t>(maxFramesInFlight * 7);

	VkDescriptorPoolCreateInfo descriptorPoolInfo{};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
	descriptorPoolInfo.pPoolSizes = descriptorPoolSizes.data();
	descriptorPoolInfo.maxSets = static_cast<uint32_t>(maxFramesInFlight * 2);

	if (vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &rayTracingAndAlphaDescriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor pool");
	}
}

void application::createComputeDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 2> poolSizes{};

	poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[0].descriptorCount = 7 * 12;

	poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[1].descriptorCount = 3 * 12;

	VkDescriptorPoolCreateInfo descriptorPoolInfo{};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	descriptorPoolInfo.pPoolSizes = poolSizes.data();
	descriptorPoolInfo.maxSets = 12;

	if (vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &computeDescriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create compute descriptor pool");
	}
}

void application::createImguiDescriptorPool()
{
	VkDescriptorPoolSize poolSizes[] = {
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo descriptorPoolInfo{};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	descriptorPoolInfo.poolSizeCount = (uint32_t)IM_ARRAYSIZE(poolSizes);
	descriptorPoolInfo.pPoolSizes = poolSizes;
	descriptorPoolInfo.maxSets = 1000 * IM_ARRAYSIZE(poolSizes);

	if (vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &imguiDescriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor pool");
	}
}

// ============================================================
// ImGui init
// ============================================================

void application::imguiInitialization()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.FontGlobalScale = 1.4f;
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(window, true);

	queueFamilyIndices queueFamily = findQueueFamilies(physicalDevice);

	ImGui_ImplVulkan_InitInfo initInfo = {};
	initInfo.Instance = instance;
	initInfo.PhysicalDevice = physicalDevice;
	initInfo.Device = device;
	initInfo.QueueFamily = queueFamily.graphicsFamily.value();
	initInfo.Queue = graphicsQueue;
	initInfo.PipelineCache = VK_NULL_HANDLE;
	initInfo.DescriptorPool = imguiDescriptorPool;
	initInfo.MinImageCount = static_cast<uint32_t>(swapChainImages.size());
	initInfo.ImageCount = static_cast<uint32_t>(swapChainImages.size());
	initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	initInfo.Allocator = nullptr;
	initInfo.RenderPass = imguiRenderPass;

	ImGui_ImplVulkan_Init(&initInfo);
}

// ============================================================
// descriptor set allocation / writing
// ============================================================

void application::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);
	VkDescriptorSetAllocateInfo descriptorSetInfo{};
	descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetInfo.descriptorPool = descriptorPool;
	descriptorSetInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	descriptorSetInfo.pSetLayouts = layouts.data();

	descriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(device, &descriptorSetInfo, descriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate descriptor sets");
	}

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		VkDescriptorBufferInfo descriptorBufferInfo{};
		descriptorBufferInfo.buffer = uniformBuffers[i];
		descriptorBufferInfo.offset = 0;
		descriptorBufferInfo.range = sizeof(uniformBufferObject);

		VkDescriptorImageInfo descriptorImageInfo{};
		descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		descriptorImageInfo.imageView = textureImageView;
		descriptorImageInfo.sampler = textureSampler;

		std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};

		writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[0].dstSet = descriptorSets[i];
		writeDescriptorSets[0].dstBinding = 0;
		writeDescriptorSets[0].dstArrayElement = 0;
		writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSets[0].descriptorCount = 1;
		writeDescriptorSets[0].pBufferInfo = &descriptorBufferInfo;
		writeDescriptorSets[0].pImageInfo = nullptr;
		writeDescriptorSets[0].pTexelBufferView = nullptr;

		writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[1].dstSet = descriptorSets[i];
		writeDescriptorSets[1].dstBinding = 1;
		writeDescriptorSets[1].dstArrayElement = 0;
		writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writeDescriptorSets[1].descriptorCount = 1;
		writeDescriptorSets[1].pBufferInfo = nullptr;
		writeDescriptorSets[1].pImageInfo = &descriptorImageInfo;
		writeDescriptorSets[1].pTexelBufferView = nullptr;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}
}

void application::createRayTracingDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, rayTracingDescriptorSetLayout);
	VkDescriptorSetAllocateInfo descriptorSetInfo{};
	descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetInfo.descriptorPool = rayTracingAndAlphaDescriptorPool;
	descriptorSetInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	descriptorSetInfo.pSetLayouts = layouts.data();

	rayTracingDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(device, &descriptorSetInfo, rayTracingDescriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate descriptor sets");
	}

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		VkDescriptorBufferInfo descriptorBufferInfo{};
		descriptorBufferInfo.buffer = uniformBuffers[i];
		descriptorBufferInfo.offset = 0;
		descriptorBufferInfo.range = sizeof(uniformBufferObject);

		VkDescriptorImageInfo storeImageInfo{};
		storeImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		storeImageInfo.imageView = storeImageView;
		storeImageInfo.sampler = VK_NULL_HANDLE;

		VkWriteDescriptorSetAccelerationStructureKHR writeAccelerationStructureSet = {};
		writeAccelerationStructureSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		writeAccelerationStructureSet.accelerationStructureCount = 1;
		writeAccelerationStructureSet.pAccelerationStructures = &tlas;

		VkDescriptorImageInfo samplerImageInfo{};
		samplerImageInfo.sampler = textureSampler;
		samplerImageInfo.imageView = textureImageView;
		samplerImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkDescriptorBufferInfo vertexBufferInfo{};
		vertexBufferInfo.buffer = vertexBuffer;
		vertexBufferInfo.offset = 0;
		vertexBufferInfo.range = sizeof(vertices[0]) * vertices.size();

		VkDescriptorBufferInfo indexBufferInfo{};
		indexBufferInfo.buffer = indexBuffer;
		indexBufferInfo.offset = 0;
		indexBufferInfo.range = sizeof(indices[0]) * indices.size();

		VkDescriptorBufferInfo sphereBufferInfo{};
		sphereBufferInfo.buffer = sphereBuffer;
		sphereBufferInfo.offset = 0;
		sphereBufferInfo.range = VK_WHOLE_SIZE;

		VkDescriptorBufferInfo materialBufferInfo{};
		materialBufferInfo.buffer = materialBuffer;
		materialBufferInfo.offset = 0;
		materialBufferInfo.range = sizeof(materials[0]) * materials.size();

		VkDescriptorBufferInfo quadBufferInfo{};
		quadBufferInfo.buffer = quadBuffer;
		quadBufferInfo.offset = 0;
		quadBufferInfo.range = VK_WHOLE_SIZE;

		VkDescriptorBufferInfo geoTypeBufferInfo{};
		geoTypeBufferInfo.buffer = geoTypeBuffer;
		geoTypeBufferInfo.offset = 0;
		geoTypeBufferInfo.range = sizeof(geoTypes[0]) * geoTypes.size();

		VkDescriptorBufferInfo aabbObjectsBufferInfo{};
		aabbObjectsBufferInfo.buffer = aabbObjectsBuffer;
		aabbObjectsBufferInfo.offset = 0;
		aabbObjectsBufferInfo.range = sizeof(gpuAabbs[0]) * gpuAabbs.size();

		VkDescriptorImageInfo normalImageInfo{};
		normalImageInfo.imageView = normalImageView;
		normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorImageInfo albedoImageInfo{};
		albedoImageInfo.imageView = albedoImageView;
		albedoImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorImageInfo velocityImageInfo{};
		velocityImageInfo.imageView = velocityImageView;
		velocityImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		std::array<VkWriteDescriptorSet, 14> writeDescriptorSets{};

		writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[0].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[0].dstBinding = 0;
		writeDescriptorSets[0].dstArrayElement = 0;
		writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		writeDescriptorSets[0].descriptorCount = 1;
		writeDescriptorSets[0].pNext = &writeAccelerationStructureSet;

		writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[1].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[1].dstBinding = 1;
		writeDescriptorSets[1].dstArrayElement = 0;
		writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeDescriptorSets[1].descriptorCount = 1;
		writeDescriptorSets[1].pImageInfo = &storeImageInfo;

		writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[2].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[2].dstBinding = 2;
		writeDescriptorSets[2].dstArrayElement = 0;
		writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writeDescriptorSets[2].descriptorCount = 1;
		writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfo;

		writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[3].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[3].dstBinding = 3;
		writeDescriptorSets[3].dstArrayElement = 0;
		writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writeDescriptorSets[3].descriptorCount = 1;
		writeDescriptorSets[3].pImageInfo = &samplerImageInfo;

		writeDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[4].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[4].dstBinding = 4;
		writeDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[4].descriptorCount = 1;
		writeDescriptorSets[4].pBufferInfo = &vertexBufferInfo;

		writeDescriptorSets[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[5].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[5].dstBinding = 5;
		writeDescriptorSets[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[5].descriptorCount = 1;
		writeDescriptorSets[5].pBufferInfo = &indexBufferInfo;

		writeDescriptorSets[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[6].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[6].dstBinding = 6;
		writeDescriptorSets[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[6].descriptorCount = 1;
		writeDescriptorSets[6].pBufferInfo = &sphereBufferInfo;

		writeDescriptorSets[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[7].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[7].dstBinding = 7;
		writeDescriptorSets[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[7].descriptorCount = 1;
		writeDescriptorSets[7].pBufferInfo = &materialBufferInfo;

		writeDescriptorSets[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[8].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[8].dstBinding = 8;
		writeDescriptorSets[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[8].descriptorCount = 1;
		writeDescriptorSets[8].pBufferInfo = &quadBufferInfo;

		writeDescriptorSets[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[9].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[9].dstBinding = 9;
		writeDescriptorSets[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[9].descriptorCount = 1;
		writeDescriptorSets[9].pBufferInfo = &geoTypeBufferInfo;

		writeDescriptorSets[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[10].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[10].dstBinding = 10;
		writeDescriptorSets[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSets[10].descriptorCount = 1;
		writeDescriptorSets[10].pBufferInfo = &aabbObjectsBufferInfo;

		writeDescriptorSets[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[11].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[11].dstBinding = 11;
		writeDescriptorSets[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeDescriptorSets[11].descriptorCount = 1;
		writeDescriptorSets[11].pImageInfo = &normalImageInfo;

		writeDescriptorSets[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[12].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[12].dstBinding = 12;
		writeDescriptorSets[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeDescriptorSets[12].descriptorCount = 1;
		writeDescriptorSets[12].pImageInfo = &albedoImageInfo;

		writeDescriptorSets[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[13].dstSet = rayTracingDescriptorSets[i];
		writeDescriptorSets[13].dstBinding = 13;
		writeDescriptorSets[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeDescriptorSets[13].descriptorCount = 1;
		writeDescriptorSets[13].pImageInfo = &velocityImageInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}
}

void application::createAlphaDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, alphaDescriptorSetLayout);
	VkDescriptorSetAllocateInfo descriptorSetInfo{};
	descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetInfo.descriptorPool = rayTracingAndAlphaDescriptorPool;
	descriptorSetInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
	descriptorSetInfo.pSetLayouts = layouts.data();

	alphaDescriptorSets.resize(maxFramesInFlight);
	if (vkAllocateDescriptorSets(device, &descriptorSetInfo, alphaDescriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate descriptor sets");
	}

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		VkDescriptorImageInfo descriptorImageInfo{};
		descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		descriptorImageInfo.imageView = alphaImageView;
		descriptorImageInfo.sampler = VK_NULL_HANDLE;

		VkDescriptorImageInfo descriptorSamplerInfo{};
		descriptorSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		descriptorSamplerInfo.imageView = VK_NULL_HANDLE;
		descriptorSamplerInfo.sampler = alphaSampler;

		std::array<VkWriteDescriptorSet, 2> writeDescriptorSets{};

		writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[0].dstSet = alphaDescriptorSets[i];
		writeDescriptorSets[0].dstBinding = 0;
		writeDescriptorSets[0].dstArrayElement = 0;
		writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		writeDescriptorSets[0].descriptorCount = 1;
		writeDescriptorSets[0].pImageInfo = &descriptorImageInfo;

		writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSets[1].dstSet = alphaDescriptorSets[i];
		writeDescriptorSets[1].dstBinding = 1;
		writeDescriptorSets[1].dstArrayElement = 0;
		writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
		writeDescriptorSets[1].descriptorCount = 1;
		writeDescriptorSets[1].pImageInfo = &descriptorSamplerInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}
}

void application::createComputeDescriptorSets()
{
	uint32_t totalSets = 12;

	std::vector<VkDescriptorSetLayout> layouts(totalSets, computeDescriptorSetLayout);

	VkDescriptorSetAllocateInfo descriptorSetInfo{};
	descriptorSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetInfo.descriptorPool = computeDescriptorPool;
	descriptorSetInfo.descriptorSetCount = totalSets;
	descriptorSetInfo.pSetLayouts = layouts.data();

	computeDescriptorSets.resize(totalSets);

	if (vkAllocateDescriptorSets(device, &descriptorSetInfo, computeDescriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate compute descriptor sets");
	}

	VkDescriptorImageInfo normalImageInfo{ VK_NULL_HANDLE, normalImageView, VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo albedoImageInfo{ VK_NULL_HANDLE, albedoImageView, VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo velocityImageInfo{ VK_NULL_HANDLE, velocityImageView, VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo prevNormalInfo{ historySampler, prevNormalImageView, VK_IMAGE_LAYOUT_GENERAL };

	VkDescriptorImageInfo storeInputInfo{ VK_NULL_HANDLE, storeImageView, VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo computeAInfo{ VK_NULL_HANDLE, computeImageViewA, VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo computeBInfo{ VK_NULL_HANDLE, computeImageViewB, VK_IMAGE_LAYOUT_GENERAL };

	auto createWrite = [](VkDescriptorSet dstSet, uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* imageInfo)
		{
			VkWriteDescriptorSet write{};
			write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			write.dstSet = dstSet;
			write.dstBinding = binding;
			write.dstArrayElement = 0;
			write.descriptorCount = 1;
			write.descriptorType = type;
			write.pImageInfo = imageInfo;
			return write;
		};

	for (uint32_t temporalParity = 0; temporalParity < 2; temporalParity++)
	{
		VkImageView readHistoryView = (temporalParity == 0) ? accumulationImageViewA : accumulationImageViewB;
		VkImageView writeHistoryView = (temporalParity == 0) ? accumulationImageViewB : accumulationImageViewA;

		VkImageView readMomentView = (temporalParity == 0) ? momentImageViewA : momentImageViewB;
		VkImageView writeMomentView = (temporalParity == 0) ? momentImageViewB : momentImageViewA;

		VkDescriptorImageInfo accumSamplerInfo{ historySampler, readHistoryView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo accumStorageInfo{ VK_NULL_HANDLE, writeHistoryView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo momentSamplerInfo{ historySampler, readMomentView, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo momentStorageInfo{ VK_NULL_HANDLE, writeMomentView, VK_IMAGE_LAYOUT_GENERAL };

		uint32_t baseSetIndex = temporalParity * 6;

		for (uint32_t pass = 0; pass < 6; pass++)
		{
			VkDescriptorSet currentSet = computeDescriptorSets[baseSetIndex + pass];

			const VkDescriptorImageInfo* inputInfo = nullptr;
			const VkDescriptorImageInfo* outputInfo = nullptr;

			if (pass == 0)
			{
				inputInfo = &storeInputInfo;
				outputInfo = &computeAInfo;
			}
			else
			{
				inputInfo = (pass % 2 == 1) ? &computeAInfo : &computeBInfo;
				outputInfo = (pass % 2 == 1) ? &computeBInfo : &computeAInfo;
			}

			std::array<VkWriteDescriptorSet, 10> writes{};
			writes[0] = createWrite(currentSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, outputInfo);
			writes[1] = createWrite(currentSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, inputInfo);
			writes[2] = createWrite(currentSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &accumSamplerInfo);
			writes[3] = createWrite(currentSet, 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumStorageInfo);
			writes[4] = createWrite(currentSet, 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normalImageInfo);
			writes[5] = createWrite(currentSet, 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &albedoImageInfo);
			writes[6] = createWrite(currentSet, 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &velocityImageInfo);
			writes[7] = createWrite(currentSet, 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prevNormalInfo);
			writes[8] = createWrite(currentSet, 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &momentSamplerInfo);
			writes[9] = createWrite(currentSet, 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &momentStorageInfo);

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		}
	}
}
