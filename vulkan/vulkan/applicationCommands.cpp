#include "Application.h"

// ============================================================
// command pool
// ============================================================

void application::createCommandPool()
{
	queueFamilyIndices queueFamily = findQueueFamilies(physicalDevice);

	VkCommandPoolCreateInfo commandPoolInfo{};
	commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolInfo.queueFamilyIndex = queueFamily.graphicsFamily.value();

	if (vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create command pool");
	}
}

// ============================================================
// command buffer allocation
// ============================================================

void application::createCommandBuffers()
{
	commandBuffers.resize(maxFramesInFlight);

	VkCommandBufferAllocateInfo commandBufferInfo{};
	commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferInfo.commandPool = commandPool;
	commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferInfo.commandBufferCount = (uint32_t)commandBuffers.size();

	if (vkAllocateCommandBuffers(device, &commandBufferInfo, commandBuffers.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate command buffers");
	}
}

// ============================================================
// single-time command helpers
// ============================================================

VkCommandBuffer application::beginSingleTimeCommands()
{
	VkCommandBufferAllocateInfo commandBufferInfo{};
	commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferInfo.commandPool = commandPool;
	commandBufferInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer);

	VkCommandBufferBeginInfo commandBufferBeginInfo{};
	commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);

	return commandBuffer;
}

void application::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// ============================================================
// command buffer recording
// ============================================================

void application::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to begin recording command buffer");
	}

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = renderPass;
	renderPassInfo.framebuffer = swapChainFrameBuffers[imageIndex];
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = swapChainExtent;

	std::array<VkClearValue, 2> clearValues{};
	clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
	clearValues[1].depthStencil = { 1.0f, 0 };
	renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderPassInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(swapChainExtent.width);
	viewport.height = static_cast<float>(swapChainExtent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = swapChainExtent;
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	VkBuffer vertexBuffers[] = { vertexBuffer };
	VkDeviceSize offsets[] = { 0 };

	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

	vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to record command buffer");
	}
}

void application::recordRayTracingCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t progressiveFrameCount, uint32_t globalFrameCount)
{
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("failed to begin recording command buffer");
	}

	vkCmdResetQueryPool(commandBuffer, timeStampQueryPools[currentFrame], 0, 6);

	VkImageSubresourceRange range{};
	range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	range.levelCount = 1;
	range.layerCount = 1;

	transitionImageLayoutInCommandBuffer(
		commandBuffer,
		swapChainImages[imageIndex],
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		range
	);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracingPipeline);

	std::array<VkDescriptorSet, 2> descriptorSetsToBind = {
		rayTracingDescriptorSets[currentFrame],
		alphaDescriptorSets[currentFrame]
	};

	vkCmdBindDescriptorSets(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
		rayTracingPipelineLayout, 0,
		static_cast<uint32_t>(descriptorSetsToBind.size()),
		descriptorSetsToBind.data(), 0, nullptr
	);

	pushConstants pcData = { progressiveFrameCount, missShaderColouring, 1, 0 };
	vkCmdPushConstants(
		commandBuffer,
		rayTracingPipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR,
		0, sizeof(pushConstants), &pcData
	);

	vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timeStampQueryPools[currentFrame], 0);

	CmdTraceRaysKHR(
		device, commandBuffer,
		&raygenRegion, &missRegion, &hitRegion, &callableRegion,
		swapChainExtent.width, swapChainExtent.height, 1
	);

	vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timeStampQueryPools[currentFrame], 1);

	std::array<VkImageMemoryBarrier, 4> rtBarriers{};

	rtBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	rtBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	rtBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
	rtBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	rtBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	rtBarriers[0].image = storeImage;
	rtBarriers[0].subresourceRange = range;

	rtBarriers[1] = rtBarriers[0];
	rtBarriers[1].image = normalImage;

	rtBarriers[2] = rtBarriers[0];
	rtBarriers[2].image = albedoImage;

	rtBarriers[3] = rtBarriers[0];
	rtBarriers[3].image = velocityImage;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr,
		static_cast<uint32_t>(rtBarriers.size()), rtBarriers.data()
	);

	VkImageCopy copyRegion{};
	copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	copyRegion.srcOffset = { 0, 0, 0 };
	copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	copyRegion.dstOffset = { 0, 0, 0 };
	copyRegion.extent = { swapChainExtent.width, swapChainExtent.height, 1 };

	transitionImageLayoutInCommandBuffer(
		commandBuffer, normalImage,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		range
	);
	transitionImageLayoutInCommandBuffer(
		commandBuffer, prevNormalImage,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		range
	);

	vkCmdCopyImage(
		commandBuffer,
		normalImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		prevNormalImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &copyRegion
	);

	transitionImageLayoutInCommandBuffer(
		commandBuffer, normalImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		range
	);
	transitionImageLayoutInCommandBuffer(
		commandBuffer, prevNormalImage,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		range
	);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

	uint32_t stepSizes[] = { 0, 1, 2, 4, 8, 16 };

	uint32_t groupCountX = (swapChainExtent.width + 15) / 16;
	uint32_t groupCountY = (swapChainExtent.height + 15) / 16;

	const uint32_t totalPasses = 6;

	uint32_t temporalParity = globalFrameCount % 2;
	uint32_t baseSetIndex = temporalParity * totalPasses;

	vkCmdWriteTimestamp(
		commandBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		timeStampQueryPools[currentFrame],
		2
	);

	VkImage readAccumImage =
		(temporalParity == 0) ? accumulationImageA : accumulationImageB;

	VkImage writeAccumImage =
		(temporalParity == 0) ? accumulationImageB : accumulationImageA;

	VkImage readMomentImage =
		(temporalParity == 0) ? momentImageA : momentImageB;

	VkImage writeMomentImage =
		(temporalParity == 0) ? momentImageB : momentImageA;

	std::array<VkImageMemoryBarrier, 4> preComputeBarriers{};

	preComputeBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	preComputeBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	preComputeBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
	preComputeBarriers[0].srcAccessMask =
		VK_ACCESS_TRANSFER_READ_BIT |
		VK_ACCESS_SHADER_READ_BIT |
		VK_ACCESS_SHADER_WRITE_BIT;
	preComputeBarriers[0].dstAccessMask =
		VK_ACCESS_SHADER_READ_BIT |
		VK_ACCESS_SHADER_WRITE_BIT;
	preComputeBarriers[0].srcQueueFamilyIndex =
		VK_QUEUE_FAMILY_IGNORED;
	preComputeBarriers[0].dstQueueFamilyIndex =
		VK_QUEUE_FAMILY_IGNORED;
	preComputeBarriers[0].image = readAccumImage;
	preComputeBarriers[0].subresourceRange =
		range;

	preComputeBarriers[1] = preComputeBarriers[0];
	preComputeBarriers[1].image = readMomentImage;

	preComputeBarriers[2] = preComputeBarriers[0];
	preComputeBarriers[2].image = computeImageA;

	preComputeBarriers[3] = preComputeBarriers[0];
	preComputeBarriers[3].image = computeImageB;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT |
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr,
		0, nullptr,
		static_cast<uint32_t>(preComputeBarriers.size()),
		preComputeBarriers.data()
	);

	for (uint32_t i = 0; i < totalPasses; ++i)
	{
		bool isTemporalPass = (i == 0);
		bool isLastPass = (i == totalPasses - 1);

		pushConstants computePc = {
			progressiveFrameCount,
			missShaderColouring,
			stepSizes[i],
			isLastPass ? 1u : 0u,
			isTemporalPass ? 1u : 0u
		};

		vkCmdPushConstants(
			commandBuffer,
			computePipelineLayout,
			VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
			0, sizeof(pushConstants), &computePc
		);

		uint32_t activeSetIndex = baseSetIndex + i;

		vkCmdBindDescriptorSets(
			commandBuffer,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			computePipelineLayout,
			0, 1,
			&computeDescriptorSets[activeSetIndex],
			0, nullptr
		);

		vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

		VkImage passOutputImage = (i % 2 == 0) ? computeImageA : computeImageB;
		VkImage passInputImage = (i % 2 == 0) ? computeImageB : computeImageA;

		if (isTemporalPass)
		{
			std::array<VkImageMemoryBarrier, 3> temporalBarriers{};

			temporalBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			temporalBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			temporalBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
			temporalBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			temporalBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			temporalBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			temporalBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			temporalBarriers[0].subresourceRange = range;

			temporalBarriers[0].image = writeAccumImage;
			temporalBarriers[1] = temporalBarriers[0];
			temporalBarriers[1].image = writeMomentImage;
			temporalBarriers[2] = temporalBarriers[0];
			temporalBarriers[2].image = computeImageA;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 0, nullptr, 0, nullptr,
				static_cast<uint32_t>(temporalBarriers.size()),
				temporalBarriers.data()
			);
		}
		else
		{
			VkImageMemoryBarrier passBarrier{};
			passBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			passBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			passBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			passBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			passBarrier.dstAccessMask = isLastPass ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
			passBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			passBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			passBarrier.image = passOutputImage;
			passBarrier.subresourceRange = range;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				isLastPass ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0, 0, nullptr, 0, nullptr,
				1, &passBarrier
			);
		}
	}

	vkCmdWriteTimestamp(
		commandBuffer,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		timeStampQueryPools[currentFrame],
		3
	);

	VkImage finalOutputImage =
		computeImageB;

	transitionImageLayoutInCommandBuffer(
		commandBuffer,
		finalOutputImage,
		VK_IMAGE_LAYOUT_GENERAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		range
	);

	VkImageBlit blitRegion{};

	blitRegion.srcSubresource = {
		VK_IMAGE_ASPECT_COLOR_BIT,
		0,
		0,
		1
	};

	blitRegion.srcOffsets[0] = {
		0, 0, 0
	};

	blitRegion.srcOffsets[1] = {
		static_cast<int32_t>(swapChainExtent.width),
		static_cast<int32_t>(swapChainExtent.height),
		1
	};

	blitRegion.dstSubresource = {
		VK_IMAGE_ASPECT_COLOR_BIT,
		0,
		0,
		1
	};

	blitRegion.dstOffsets[0] = {
		0, 0, 0
	};

	blitRegion.dstOffsets[1] = {
		static_cast<int32_t>(swapChainExtent.width),
		static_cast<int32_t>(swapChainExtent.height),
		1
	};

	vkCmdBlitImage(
		commandBuffer,
		finalOutputImage,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		swapChainImages[imageIndex],
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&blitRegion,
		VK_FILTER_NEAREST
	);

	VkImageMemoryBarrier backToGeneralBarrier{};

	backToGeneralBarrier.sType =
		VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

	backToGeneralBarrier.oldLayout =
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

	backToGeneralBarrier.newLayout =
		VK_IMAGE_LAYOUT_GENERAL;

	backToGeneralBarrier.srcQueueFamilyIndex =
		VK_QUEUE_FAMILY_IGNORED;

	backToGeneralBarrier.dstQueueFamilyIndex =
		VK_QUEUE_FAMILY_IGNORED;

	backToGeneralBarrier.image =
		finalOutputImage;

	backToGeneralBarrier.subresourceRange =
		range;

	backToGeneralBarrier.srcAccessMask =
		VK_ACCESS_TRANSFER_READ_BIT;

	backToGeneralBarrier.dstAccessMask =
		VK_ACCESS_SHADER_READ_BIT |
		VK_ACCESS_SHADER_WRITE_BIT;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr,
		0, nullptr,
		1,
		&backToGeneralBarrier
	);

	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType =
		VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

	renderPassInfo.renderPass =
		imguiRenderPass;

	renderPassInfo.framebuffer =
		imguiFrameBuffers[imageIndex];

	renderPassInfo.renderArea.offset =
	{ 0, 0 };

	renderPassInfo.renderArea.extent =
		swapChainExtent;

	VkClearValue clearValue{};
	clearValue.color =
	{ {0.0f, 0.0f, 0.0f, 1.0f} };

	renderPassInfo.clearValueCount =
		1;

	renderPassInfo.pClearValues =
		&clearValue;

	vkCmdWriteTimestamp(
		commandBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		timeStampQueryPools[currentFrame],
		4
	);

	vkCmdBeginRenderPass(
		commandBuffer,
		&renderPassInfo,
		VK_SUBPASS_CONTENTS_INLINE
	);

	ImGui_ImplVulkan_RenderDrawData(
		ImGui::GetDrawData(),
		commandBuffer
	);

	vkCmdEndRenderPass(commandBuffer);

	vkCmdWriteTimestamp(
		commandBuffer,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		timeStampQueryPools[currentFrame],
		5
	);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to record command buffer");
	}
}