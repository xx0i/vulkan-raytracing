#include "Application.h"

// ============================================================
// pipelines
// ============================================================

void application::createGraphicsPipeline()
{
	auto vertexShaderCode = readFile("shaders/vert.spv");
	auto fragmentShaderCode = readFile("shaders/frag.spv");

	VkShaderModule vertexShaderModule = createShaderModule(vertexShaderCode);
	VkShaderModule fragmentShaderModule = createShaderModule(fragmentShaderCode);

	VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
	vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertexShaderStageInfo.module = vertexShaderModule;
	vertexShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
	fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragmentShaderStageInfo.module = fragmentShaderModule;
	fragmentShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertexShaderStageInfo, fragmentShaderStageInfo };

	auto bindingDescription = vertex::getBindingDescription();
	auto attributeDescriptions = vertex::getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo{};
	inputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;

	VkPipelineViewportStateCreateInfo viewportStateInfo{};
	viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateInfo.viewportCount = 1;
	viewportStateInfo.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo{};
	rasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationStateInfo.depthClampEnable = VK_FALSE;
	rasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationStateInfo.lineWidth = 1.0f;
	rasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationStateInfo.depthBiasEnable = VK_FALSE;
	rasterizationStateInfo.depthBiasConstantFactor = 0.0f;
	rasterizationStateInfo.depthBiasClamp = 0.0f;
	rasterizationStateInfo.depthBiasSlopeFactor = 0.0f;

	VkPipelineMultisampleStateCreateInfo multisampleStateInfo{};
	multisampleStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleStateInfo.sampleShadingEnable = VK_TRUE;
	multisampleStateInfo.rasterizationSamples = msaaSamples;
	multisampleStateInfo.minSampleShading = 0.2f;
	multisampleStateInfo.pSampleMask = nullptr;
	multisampleStateInfo.alphaToCoverageEnable = VK_FALSE;
	multisampleStateInfo.alphaToOneEnable = VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo{};
	depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilStateInfo.depthTestEnable = VK_TRUE;
	depthStencilStateInfo.depthWriteEnable = VK_TRUE;
	depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilStateInfo.depthBoundsTestEnable = VK_FALSE;
	depthStencilStateInfo.minDepthBounds = 0.0f;
	depthStencilStateInfo.maxDepthBounds = 1.0f;
	depthStencilStateInfo.stencilTestEnable = VK_FALSE;
	depthStencilStateInfo.front = {};
	depthStencilStateInfo.back = {};

	VkPipelineColorBlendAttachmentState colorBlendAttatchmentState{};
	colorBlendAttatchmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttatchmentState.blendEnable = VK_FALSE;
	colorBlendAttatchmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttatchmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttatchmentState.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttatchmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttatchmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttatchmentState.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlendStateInfo{};
	colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendStateInfo.logicOpEnable = VK_FALSE;
	colorBlendStateInfo.logicOp = VK_LOGIC_OP_COPY;
	colorBlendStateInfo.attachmentCount = 1;
	colorBlendStateInfo.pAttachments = &colorBlendAttatchmentState;
	colorBlendStateInfo.blendConstants[0] = 0.0f;
	colorBlendStateInfo.blendConstants[1] = 0.0f;
	colorBlendStateInfo.blendConstants[2] = 0.0f;
	colorBlendStateInfo.blendConstants[3] = 0.0f;

	std::vector<VkDynamicState> dynamicStates =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
	dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicStateInfo.pDynamicStates = dynamicStates.data();

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;

	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create pipeline layout");
	}

	VkGraphicsPipelineCreateInfo graphicsPipelineInfo{};
	graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	graphicsPipelineInfo.stageCount = 2;
	graphicsPipelineInfo.pStages = shaderStages;
	graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
	graphicsPipelineInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
	graphicsPipelineInfo.pViewportState = &viewportStateInfo;
	graphicsPipelineInfo.pRasterizationState = &rasterizationStateInfo;
	graphicsPipelineInfo.pMultisampleState = &multisampleStateInfo;
	graphicsPipelineInfo.pDepthStencilState = &depthStencilStateInfo;
	graphicsPipelineInfo.pColorBlendState = &colorBlendStateInfo;
	graphicsPipelineInfo.pDynamicState = &dynamicStateInfo;
	graphicsPipelineInfo.layout = pipelineLayout;
	graphicsPipelineInfo.renderPass = renderPass;
	graphicsPipelineInfo.subpass = 0;
	graphicsPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
	graphicsPipelineInfo.basePipelineIndex = -1;

	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create graphics pipeline");
	}

	vkDestroyShaderModule(device, fragmentShaderModule, nullptr);
	vkDestroyShaderModule(device, vertexShaderModule, nullptr);
}

void application::createRayTracingPipeline()
{
	auto rayGenShaderCode = readFile("rayTracing/rgen.spv");
	auto missShaderCode = readFile("rayTracing/rmiss.spv");
	auto shadowMissShaderCode = readFile("rayTracing/shadowrmiss.spv");
	auto closesthitShaderCode = readFile("rayTracing/rchit.spv");
	auto quadClosesthitShaderCode = readFile("rayTracing/quadrchit.spv");
	auto sphereClosesthitShaderCode = readFile("rayTracing/sphererchit.spv");
	auto anyhitShaderCode = readFile("rayTracing/rahit.spv");
	auto intersectionCode = readFile("rayTracing/rint.spv");
	auto quadIntersectionCode = readFile("rayTracing/quad.spv");
	auto sphereIntersectionCode = readFile("rayTracing/sphere.spv");

	VkShaderModule rayGenShaderModule = createShaderModule(rayGenShaderCode);
	VkShaderModule missShaderModule = createShaderModule(missShaderCode);
	VkShaderModule shadowMissShaderModule = createShaderModule(shadowMissShaderCode);
	VkShaderModule closesthitShaderModule = createShaderModule(closesthitShaderCode);
	VkShaderModule quadClosesthitShaderModule = createShaderModule(quadClosesthitShaderCode);
	VkShaderModule sphereClosesthitShaderModule = createShaderModule(sphereClosesthitShaderCode);
	VkShaderModule anyhitShaderModule = createShaderModule(anyhitShaderCode);
	VkShaderModule intersectionShaderModule = createShaderModule(intersectionCode);
	VkShaderModule quadIntersectionShaderModule = createShaderModule(quadIntersectionCode);
	VkShaderModule sphereIntersectionShaderModule = createShaderModule(sphereIntersectionCode);

	VkPipelineShaderStageCreateInfo rayGenShaderStageInfo{};
	rayGenShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	rayGenShaderStageInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	rayGenShaderStageInfo.module = rayGenShaderModule;
	rayGenShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo missShaderStageInfo{};
	missShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	missShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	missShaderStageInfo.module = missShaderModule;
	missShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shadowMissShaderStageInfo{};
	shadowMissShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shadowMissShaderStageInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	shadowMissShaderStageInfo.module = shadowMissShaderModule;
	shadowMissShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo closesthitShaderStageInfo{};
	closesthitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	closesthitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	closesthitShaderStageInfo.module = closesthitShaderModule;
	closesthitShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo quadClosesthitShaderStageInfo{};
	quadClosesthitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	quadClosesthitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	quadClosesthitShaderStageInfo.module = quadClosesthitShaderModule;
	quadClosesthitShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo sphereClosesthitShaderStageInfo{};
	sphereClosesthitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	sphereClosesthitShaderStageInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	sphereClosesthitShaderStageInfo.module = sphereClosesthitShaderModule;
	sphereClosesthitShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo anyhitShaderStageInfo{};
	anyhitShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	anyhitShaderStageInfo.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
	anyhitShaderStageInfo.module = anyhitShaderModule;
	anyhitShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo intersectionShaderStageInfo{};
	intersectionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	intersectionShaderStageInfo.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	intersectionShaderStageInfo.module = intersectionShaderModule;
	intersectionShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo quadIntersectionShaderStageInfo{};
	quadIntersectionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	quadIntersectionShaderStageInfo.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	quadIntersectionShaderStageInfo.module = quadIntersectionShaderModule;
	quadIntersectionShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo sphereIntersectionShaderStageInfo{};
	sphereIntersectionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	sphereIntersectionShaderStageInfo.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	sphereIntersectionShaderStageInfo.module = sphereIntersectionShaderModule;
	sphereIntersectionShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { rayGenShaderStageInfo, missShaderStageInfo, shadowMissShaderStageInfo, closesthitShaderStageInfo,
														quadClosesthitShaderStageInfo, sphereClosesthitShaderStageInfo,	anyhitShaderStageInfo,
														intersectionShaderStageInfo, quadIntersectionShaderStageInfo, sphereIntersectionShaderStageInfo };

	VkRayTracingShaderGroupCreateInfoKHR rayGenStage{};
	rayGenStage.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	rayGenStage.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	rayGenStage.generalShader = 0;
	rayGenStage.closestHitShader = VK_SHADER_UNUSED_KHR;
	rayGenStage.anyHitShader = VK_SHADER_UNUSED_KHR;
	rayGenStage.intersectionShader = VK_SHADER_UNUSED_KHR;

	VkRayTracingShaderGroupCreateInfoKHR missStage{};
	missStage.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	missStage.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	missStage.generalShader = 1;
	missStage.closestHitShader = VK_SHADER_UNUSED_KHR;
	missStage.anyHitShader = VK_SHADER_UNUSED_KHR;
	missStage.intersectionShader = VK_SHADER_UNUSED_KHR;

	VkRayTracingShaderGroupCreateInfoKHR shadowMissStage{};
	shadowMissStage.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	shadowMissStage.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	shadowMissStage.generalShader = 2;
	shadowMissStage.closestHitShader = VK_SHADER_UNUSED_KHR;
	shadowMissStage.anyHitShader = VK_SHADER_UNUSED_KHR;
	shadowMissStage.intersectionShader = VK_SHADER_UNUSED_KHR;

	VkRayTracingShaderGroupCreateInfoKHR hitStages{};
	hitStages.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	hitStages.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
	hitStages.generalShader = VK_SHADER_UNUSED_KHR;
	hitStages.closestHitShader = 3;
	hitStages.anyHitShader = 6;
	hitStages.intersectionShader = 7;

	VkRayTracingShaderGroupCreateInfoKHR quadHitStage{};
	quadHitStage.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	quadHitStage.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
	quadHitStage.generalShader = VK_SHADER_UNUSED_KHR;
	quadHitStage.closestHitShader = 4;
	quadHitStage.anyHitShader = 6;
	quadHitStage.intersectionShader = 8;

	VkRayTracingShaderGroupCreateInfoKHR sphereHitStage{};
	sphereHitStage.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	sphereHitStage.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
	sphereHitStage.generalShader = VK_SHADER_UNUSED_KHR;
	sphereHitStage.closestHitShader = 5;
	sphereHitStage.anyHitShader = 6;
	sphereHitStage.intersectionShader = 9;

	VkRayTracingShaderGroupCreateInfoKHR shaderGroups[] = { rayGenStage, missStage, shadowMissStage, hitStages, quadHitStage, sphereHitStage };

	std::array<VkDescriptorSetLayout, 2> layouts = { rayTracingDescriptorSetLayout, alphaDescriptorSetLayout };

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(pushConstants);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
	pipelineLayoutInfo.pSetLayouts = layouts.data();
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &rayTracingPipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create ray tracing pipeline layout");
	}

	VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	pipelineInfo.stageCount = static_cast<uint32_t>(std::size(shaderStages));
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.groupCount = static_cast<uint32_t>(std::size(shaderGroups));
	pipelineInfo.pGroups = shaderGroups;
	pipelineInfo.maxPipelineRayRecursionDepth = 2;
	pipelineInfo.layout = rayTracingPipelineLayout;

	CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &rayTracingPipeline);

	vkDestroyShaderModule(device, rayGenShaderModule, nullptr);
	vkDestroyShaderModule(device, missShaderModule, nullptr);
	vkDestroyShaderModule(device, shadowMissShaderModule, nullptr);
	vkDestroyShaderModule(device, closesthitShaderModule, nullptr);
	vkDestroyShaderModule(device, quadClosesthitShaderModule, nullptr);
	vkDestroyShaderModule(device, sphereClosesthitShaderModule, nullptr);
	vkDestroyShaderModule(device, anyhitShaderModule, nullptr);
	vkDestroyShaderModule(device, intersectionShaderModule, nullptr);
	vkDestroyShaderModule(device, quadIntersectionShaderModule, nullptr);
	vkDestroyShaderModule(device, sphereIntersectionShaderModule, nullptr);
}

void application::createComputePipeline()
{
	auto computeShaderCode = readFile("rayTracing/comp.spv");

	VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

	VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
	computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computeShaderStageInfo.module = computeShaderModule;
	computeShaderStageInfo.pName = "main";

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(pushConstants);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create compute pipeline layout");
	}

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = computeShaderStageInfo;
	pipelineInfo.layout = computePipelineLayout;

	vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);

	vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

void application::createShaderBindingTables()
{
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
	rayTracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	VkPhysicalDeviceProperties2 deviceProperties2{};
	deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties2.pNext = &rayTracingProperties;
	vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);
	uint32_t handleSize = rayTracingProperties.shaderGroupHandleSize;
	uint32_t handleAlignment = (handleSize + rayTracingProperties.shaderGroupBaseAlignment - 1) & ~(rayTracingProperties.shaderGroupBaseAlignment - 1);

	uint32_t groupCount = 6;

	uint32_t sbtSize = groupCount * handleAlignment;
	std::vector<uint8_t> shaderHandleStorage(sbtSize);
	GetRayTracingShaderGroupHandlesKHR(device, rayTracingPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());
	createBuffer(sbtSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		shaderBindingTableBuffer, shaderBindingTableBufferMemory);
	void* data;
	vkMapMemory(device, shaderBindingTableBufferMemory, 0, sbtSize, 0, &data);
	for (uint32_t i = 0; i < groupCount; i++)
	{
		memcpy((char*)data + i * handleAlignment, shaderHandleStorage.data() + i * handleSize, handleSize);
	}
	vkUnmapMemory(device, shaderBindingTableBufferMemory);
	shaderBindingTableAddress = findBufferDeviceAddress(device, shaderBindingTableBuffer);

	raygenRegion.deviceAddress = shaderBindingTableAddress + 0 * handleAlignment;
	raygenRegion.stride = handleAlignment;
	raygenRegion.size = handleAlignment;

	missRegion.deviceAddress = shaderBindingTableAddress + 1 * handleAlignment;
	missRegion.stride = handleAlignment;
	missRegion.size = 2 * handleAlignment;

	hitRegion.deviceAddress = shaderBindingTableAddress + 3 * handleAlignment;
	hitRegion.stride = handleAlignment;
	hitRegion.size = 3 * handleAlignment;

	callableRegion.deviceAddress = 0;
	callableRegion.stride = 0;
	callableRegion.size = 0;
}
