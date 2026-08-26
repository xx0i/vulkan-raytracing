#include "Application.h"

// ============================================================
// descriptor set layouts
// ============================================================

void application::createDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	uboLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 1;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.pImmutableSamplers = nullptr;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
	VkDescriptorSetLayoutCreateInfo  layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor set layout");
	}
}

void application::createRayTracingDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding topLevelASBinding{};
	topLevelASBinding.binding = 0;
	topLevelASBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	topLevelASBinding.descriptorCount = 1;
	topLevelASBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	topLevelASBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding outputImageLayoutBinding{};
	outputImageLayoutBinding.binding = 1;
	outputImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	outputImageLayoutBinding.descriptorCount = 1;
	outputImageLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	outputImageLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 2;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	uboLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding texSamplerLayoutBinding{};
	texSamplerLayoutBinding.binding = 3;
	texSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	texSamplerLayoutBinding.descriptorCount = 1;
	texSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	texSamplerLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding vertexBinding{};
	vertexBinding.binding = 4;
	vertexBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vertexBinding.descriptorCount = 1;
	vertexBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding indexBinding{};
	indexBinding.binding = 5;
	indexBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	indexBinding.descriptorCount = 1;
	indexBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding sphereBinding{};
	sphereBinding.binding = 6;
	sphereBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	sphereBinding.descriptorCount = 1;
	sphereBinding.stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding materialBinding{};
	materialBinding.binding = 7;
	materialBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	materialBinding.descriptorCount = 1;
	materialBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding quadBinding{};
	quadBinding.binding = 8;
	quadBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	quadBinding.descriptorCount = 1;
	quadBinding.stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding geoTypeBinding{};
	geoTypeBinding.binding = 9;
	geoTypeBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	geoTypeBinding.descriptorCount = 1;
	geoTypeBinding.stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding aabbObjectsBinding{};
	aabbObjectsBinding.binding = 10;
	aabbObjectsBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	aabbObjectsBinding.descriptorCount = 1;
	aabbObjectsBinding.stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	VkDescriptorSetLayoutBinding gbufferNormalBinding{};
	gbufferNormalBinding.binding = 11;
	gbufferNormalBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	gbufferNormalBinding.descriptorCount = 1;
	gbufferNormalBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	gbufferNormalBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding gbufferAlbedoBinding{};
	gbufferAlbedoBinding.binding = 12;
	gbufferAlbedoBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	gbufferAlbedoBinding.descriptorCount = 1;
	gbufferAlbedoBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	gbufferAlbedoBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding gbufferVelocityBinding{};
	gbufferVelocityBinding.binding = 13;
	gbufferVelocityBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	gbufferVelocityBinding.descriptorCount = 1;
	gbufferVelocityBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	gbufferVelocityBinding.pImmutableSamplers = nullptr;


	std::array<VkDescriptorSetLayoutBinding, 14> bindings =
	{
		topLevelASBinding, outputImageLayoutBinding, uboLayoutBinding, texSamplerLayoutBinding, vertexBinding, indexBinding, sphereBinding,
		materialBinding, quadBinding, geoTypeBinding, aabbObjectsBinding, gbufferNormalBinding, gbufferAlbedoBinding, gbufferVelocityBinding
	};

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &rayTracingDescriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor set layout");
	}
}

void application::createAlphaDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding alphaImageBinding{};
	alphaImageBinding.binding = 0;
	alphaImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	alphaImageBinding.descriptorCount = 1;
	alphaImageBinding.stageFlags = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
	alphaImageBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding alphaSamplerBinding = {};
	alphaSamplerBinding.binding = 1;
	alphaSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
	alphaSamplerBinding.descriptorCount = 1;
	alphaSamplerBinding.stageFlags = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = { alphaImageBinding, alphaSamplerBinding };

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &alphaDescriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create alpha descriptor set layout");
	}
}

void application::createComputeDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding outputImageBinding{};
	outputImageBinding.binding = 0;
	outputImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	outputImageBinding.descriptorCount = 1;
	outputImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding inputImageBinding{};
	inputImageBinding.binding = 1;
	inputImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	inputImageBinding.descriptorCount = 1;
	inputImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding accumulationSamplerBinding{};
	accumulationSamplerBinding.binding = 2;
	accumulationSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	accumulationSamplerBinding.descriptorCount = 1;
	accumulationSamplerBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding accumulationStorageBinding{};
	accumulationStorageBinding.binding = 3;
	accumulationStorageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	accumulationStorageBinding.descriptorCount = 1;
	accumulationStorageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding gbufferNormalBinding{};
	gbufferNormalBinding.binding = 4;
	gbufferNormalBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	gbufferNormalBinding.descriptorCount = 1;
	gbufferNormalBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding gbufferAlbedoBinding{};
	gbufferAlbedoBinding.binding = 5;
	gbufferAlbedoBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	gbufferAlbedoBinding.descriptorCount = 1;
	gbufferAlbedoBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding gbufferVelocityBinding{};
	gbufferVelocityBinding.binding = 6;
	gbufferVelocityBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	gbufferVelocityBinding.descriptorCount = 1;
	gbufferVelocityBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding prevNormalDepthBinding{};
	prevNormalDepthBinding.binding = 7;
	prevNormalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	prevNormalDepthBinding.descriptorCount = 1;
	prevNormalDepthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding prevMomentsSamplerBinding{};
	prevMomentsSamplerBinding.binding = 8;
	prevMomentsSamplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	prevMomentsSamplerBinding.descriptorCount = 1;
	prevMomentsSamplerBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutBinding momentsStorageBinding{};
	momentsStorageBinding.binding = 9;
	momentsStorageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	momentsStorageBinding.descriptorCount = 1;
	momentsStorageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	std::array<VkDescriptorSetLayoutBinding, 10> bindings = {
		outputImageBinding, inputImageBinding, accumulationSamplerBinding, accumulationStorageBinding,
		gbufferNormalBinding, gbufferAlbedoBinding, gbufferVelocityBinding, prevNormalDepthBinding,
		prevMomentsSamplerBinding, momentsStorageBinding
	};

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create compute descriptor set layout");
	}
}