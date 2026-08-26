#include "Application.h"

// ============================================================
// acceleration structures
// ============================================================

void application::createAccerlerationStructures()
{
	for (size_t i = 0; i < aabbObjects.size(); i++)
	{
		aabbObjects[i].blas = createBLASForAABB(aabbObjects[i].aabb, aabbObjects[i].blasDeviceAddress,
			aabbObjects[i].blasBuffer, aabbObjects[i].blasMemory);
	}

	for (auto& obj : aabbObjects) {
		std::cout << "Object type: " << int(obj.type)
			<< " BLAS device address: " << obj.blasDeviceAddress
			<< " AABB min: " << obj.aabb.minX << "," << obj.aabb.minY << "," << obj.aabb.minZ
			<< " max: " << obj.aabb.maxX << "," << obj.aabb.maxY << "," << obj.aabb.maxZ << "\n";
	}

	createMultiInstanceTLAS();
}

void application::createBLAStriangle()
{
	VkAccelerationStructureGeometryTrianglesDataKHR triangleData{};
	triangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	triangleData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	triangleData.vertexData.deviceAddress = vertexAddress;
	triangleData.vertexStride = sizeof(vertex);
	triangleData.indexType = VK_INDEX_TYPE_UINT32;
	triangleData.indexData.deviceAddress = indexAddress;
	triangleData.maxVertex = static_cast<uint32_t>(vertices.size());

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.triangles = triangleData;

	VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
	buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	buildGeometryInfo.geometryCount = 1;
	buildGeometryInfo.pGeometries = &geometry;
	buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

	VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {};
	buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

	uint32_t maxPrimitiveCount = static_cast<uint32_t>(indices.size()) / 3;

	GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo, &maxPrimitiveCount, &buildSizesInfo);

	createBuffer(buildSizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blasBuffer, blasMemory);

	VkAccelerationStructureCreateInfoKHR blasCreateInfo = {};
	blasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	blasCreateInfo.size = buildSizesInfo.accelerationStructureSize;
	blasCreateInfo.buffer = blasBuffer;

	CreateAccelerationStructureKHR(device, &blasCreateInfo, nullptr, &blas);

	buildGeometryInfo.dstAccelerationStructure = blas;

	VkBuffer scratchBuffer;
	VkDeviceMemory scratchMemory;

	createBuffer(buildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

	VkDeviceAddress scratchAddress = findBufferDeviceAddress(device, scratchBuffer);
	buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
	buildRangeInfo.primitiveCount = maxPrimitiveCount;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.transformOffset = 0;

	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = &buildRangeInfo;

	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	CmdBuildAccelerationStructuresKHR(device, commandBuffer, 1, &buildGeometryInfo, &rangeInfo);

	endSingleTimeCommands(commandBuffer);

	vkFreeMemory(device, scratchMemory, nullptr);
	vkDestroyBuffer(device, scratchBuffer, nullptr);
}

void application::createBLASaabb()
{
	VkAccelerationStructureGeometryAabbsDataKHR aabbData{};
	aabbData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
	aabbData.data.deviceAddress = aabbAddress;
	aabbData.stride = sizeof(VkAabbPositionsKHR);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.aabbs = aabbData;

	VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
	buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	buildGeometryInfo.geometryCount = 1;
	buildGeometryInfo.pGeometries = &geometry;
	buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

	VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {};
	buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

	uint32_t maxPrimitiveCount = static_cast<uint32_t>(aabbs.size());

	GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo, &maxPrimitiveCount, &buildSizesInfo);

	createBuffer(buildSizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blasBuffer, blasMemory);

	VkAccelerationStructureCreateInfoKHR blasCreateInfo = {};
	blasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	blasCreateInfo.size = buildSizesInfo.accelerationStructureSize;
	blasCreateInfo.buffer = blasBuffer;

	CreateAccelerationStructureKHR(device, &blasCreateInfo, nullptr, &blas);

	buildGeometryInfo.dstAccelerationStructure = blas;

	VkBuffer scratchBuffer;
	VkDeviceMemory scratchMemory;

	createBuffer(buildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

	VkDeviceAddress scratchAddress = findBufferDeviceAddress(device, scratchBuffer);
	buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
	buildRangeInfo.primitiveCount = maxPrimitiveCount;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.transformOffset = 0;

	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = &buildRangeInfo;

	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	CmdBuildAccelerationStructuresKHR(device, commandBuffer, 1, &buildGeometryInfo, &rangeInfo);

	endSingleTimeCommands(commandBuffer);

	vkFreeMemory(device, scratchMemory, nullptr);
	vkDestroyBuffer(device, scratchBuffer, nullptr);
}

VkAccelerationStructureKHR application::createBLASForAABB(const VkAabbPositionsKHR& aabb, VkDeviceAddress& deviceAddr, VkBuffer& buffer, VkDeviceMemory& memory)
{
	VkBuffer aabbUploadBuffer;
	VkDeviceMemory aabbUploadMemory;
	VkDeviceSize aabbSize = sizeof(VkAabbPositionsKHR);

	createBuffer(aabbSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, aabbUploadBuffer, aabbUploadMemory);

	void* mapped;
	vkMapMemory(device, aabbUploadMemory, 0, aabbSize, 0, &mapped);
	memcpy(mapped, &aabb, sizeof(VkAabbPositionsKHR));
	vkUnmapMemory(device, aabbUploadMemory);

	VkDeviceAddress aabbDeviceAddress = findBufferDeviceAddress(device, aabbUploadBuffer);

	VkAccelerationStructureGeometryAabbsDataKHR aabbData{};
	aabbData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
	aabbData.data.deviceAddress = aabbDeviceAddress;
	aabbData.stride = sizeof(VkAabbPositionsKHR);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.aabbs = aabbData;

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
	buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &geometry;
	buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
	sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

	uint32_t primitiveCount = 1;
	GetAccelerationStructureBuildSizesKHR(device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo,
		&primitiveCount,
		&sizeInfo);

	VkBuffer blasBuffer;
	VkDeviceMemory blasMemory;

	createBuffer(sizeInfo.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		blasBuffer, blasMemory);

	VkAccelerationStructureCreateInfoKHR blasCreateInfo{};
	blasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	blasCreateInfo.size = sizeInfo.accelerationStructureSize;
	blasCreateInfo.buffer = blasBuffer;

	VkAccelerationStructureKHR blas;
	CreateAccelerationStructureKHR(device, &blasCreateInfo, nullptr, &blas);

	buildInfo.dstAccelerationStructure = blas;

	VkBuffer scratchBuffer;
	VkDeviceMemory scratchMemory;
	createBuffer(sizeInfo.buildScratchSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		scratchBuffer, scratchMemory);

	VkDeviceAddress scratchAddress = findBufferDeviceAddress(device, scratchBuffer);
	buildInfo.scratchData.deviceAddress = scratchAddress;

	VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
	rangeInfo.primitiveCount = primitiveCount;
	rangeInfo.primitiveOffset = 0;
	rangeInfo.firstVertex = 0;
	rangeInfo.transformOffset = 0;
	const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

	VkCommandBuffer cmd = beginSingleTimeCommands();
	CmdBuildAccelerationStructuresKHR(device, cmd, 1, &buildInfo, &pRangeInfo);
	endSingleTimeCommands(cmd);

	vkFreeMemory(device, scratchMemory, nullptr);
	vkDestroyBuffer(device, scratchBuffer, nullptr);

	vkFreeMemory(device, aabbUploadMemory, nullptr);
	vkDestroyBuffer(device, aabbUploadBuffer, nullptr);

	deviceAddr = findAccelerationStructureDeviceAddress(device, blas);
	buffer = blasBuffer;
	memory = blasMemory;

	return blas;
}

void application::createTLAS()
{
	VkAccelerationStructureInstanceKHR asInstance{};

	VkTransformMatrixKHR identityTransform = {
	{ {1.0f, 0.0f, 0.0f, 0.0f},
	{0.0f, 1.0f, 0.0f, 0.0f},
	{0.0f, 0.0f, 1.0f, 0.0f} }
	};
	asInstance.transform = identityTransform;

	asInstance.instanceCustomIndex = 0;
	asInstance.mask = 0xFF;
	asInstance.instanceShaderBindingTableRecordOffset = 0;
	asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	asInstance.accelerationStructureReference = findAccelerationStructureDeviceAddress(device, blas);

	VkBuffer instanceBuffer;
	VkDeviceMemory instanceMemory;
	createBuffer(sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, instanceBuffer, instanceMemory);

	void* data;
	vkMapMemory(device, instanceMemory, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &data);
	memcpy(data, &asInstance, sizeof(VkAccelerationStructureInstanceKHR));
	vkUnmapMemory(device, instanceMemory);

	VkDeviceAddress instanceAddress = findBufferDeviceAddress(device, instanceBuffer);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geometry.geometry.instances.arrayOfPointers = VK_FALSE;
	geometry.geometry.instances.data.deviceAddress = instanceAddress;

	VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
	buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildGeometryInfo.geometryCount = 1;
	buildGeometryInfo.pGeometries = &geometry;
	buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

	VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {};
	buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

	uint32_t maxPrimitiveCount = 1;

	GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo, &maxPrimitiveCount, &buildSizesInfo);

	createBuffer(buildSizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		tlasBuffer, tlasMemory);

	VkAccelerationStructureCreateInfoKHR tlasCreateInfo = {};
	tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	tlasCreateInfo.size = buildSizesInfo.accelerationStructureSize;
	tlasCreateInfo.buffer = tlasBuffer;

	CreateAccelerationStructureKHR(device, &tlasCreateInfo, nullptr, &tlas);

	buildGeometryInfo.dstAccelerationStructure = tlas;

	VkBuffer scratchBuffer;
	VkDeviceMemory scratchMemory;

	createBuffer(buildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

	VkDeviceAddress scratchAddress = findBufferDeviceAddress(device, scratchBuffer);
	buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
	buildRangeInfo.primitiveCount = maxPrimitiveCount;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.transformOffset = 0;

	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = &buildRangeInfo;

	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	CmdBuildAccelerationStructuresKHR(device, commandBuffer, 1, &buildGeometryInfo, &rangeInfo);

	endSingleTimeCommands(commandBuffer);

	vkFreeMemory(device, scratchMemory, nullptr);
	vkDestroyBuffer(device, scratchBuffer, nullptr);
	vkFreeMemory(device, instanceMemory, nullptr);
	vkDestroyBuffer(device, instanceBuffer, nullptr);
}

void application::createMultiInstanceTLAS()
{
	std::vector<VkAccelerationStructureInstanceKHR> instances{};
	instances.reserve(aabbObjects.size());

	for (size_t i = 0; i < aabbObjects.size(); i++)
	{
		const auto& obj = aabbObjects[i];

		VkAccelerationStructureInstanceKHR instance{};
		VkTransformMatrixKHR identityTransform = {
		{{1.0f, 0.0f, 0.0f, 0.0f},
		{0.0f, 1.0f, 0.0f, 0.0f},
		{0.0f, 0.0f, 1.0f, 0.0f}}
		};
		instance.transform = identityTransform;

		instance.instanceCustomIndex = static_cast<uint32_t>(i);
		instance.mask = 0xFF;

		if (obj.type == geometryType::quadShape)
		{
			instance.instanceShaderBindingTableRecordOffset = 1;
		}
		else if (obj.type == geometryType::sphereShape)
		{
			instance.instanceShaderBindingTableRecordOffset = 2;
		}
		else
		{
			instance.instanceShaderBindingTableRecordOffset = 0;
		}

		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instance.accelerationStructureReference = obj.blasDeviceAddress;

		instances.push_back(instance);
	}

	VkBuffer instanceBuffer;
	VkDeviceMemory instanceMemory;
	VkDeviceSize bufferSize = sizeof(VkAccelerationStructureInstanceKHR) * instances.size();
	createBuffer(bufferSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		instanceBuffer, instanceMemory);

	void* data;
	vkMapMemory(device, instanceMemory, 0, bufferSize, 0, &data);
	memcpy(data, instances.data(), bufferSize);
	vkUnmapMemory(device, instanceMemory);

	VkDeviceAddress instanceAddress = findBufferDeviceAddress(device, instanceBuffer);

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geometry.geometry.instances.arrayOfPointers = VK_FALSE;
	geometry.geometry.instances.data.deviceAddress = instanceAddress;

	VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{};
	buildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildGeometryInfo.geometryCount = 1;
	buildGeometryInfo.pGeometries = &geometry;
	buildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

	VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {};
	buildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

	uint32_t maxPrimitiveCount = static_cast<uint32_t>(instances.size());

	GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo, &maxPrimitiveCount, &buildSizesInfo);

	createBuffer(buildSizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		tlasBuffer, tlasMemory);

	VkAccelerationStructureCreateInfoKHR tlasCreateInfo = {};
	tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	tlasCreateInfo.size = buildSizesInfo.accelerationStructureSize;
	tlasCreateInfo.buffer = tlasBuffer;

	CreateAccelerationStructureKHR(device, &tlasCreateInfo, nullptr, &tlas);

	buildGeometryInfo.dstAccelerationStructure = tlas;

	VkBuffer scratchBuffer;
	VkDeviceMemory scratchMemory;

	createBuffer(buildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

	VkDeviceAddress scratchAddress = findBufferDeviceAddress(device, scratchBuffer);
	buildGeometryInfo.scratchData.deviceAddress = scratchAddress;

	VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
	buildRangeInfo.primitiveCount = maxPrimitiveCount;
	buildRangeInfo.primitiveOffset = 0;
	buildRangeInfo.firstVertex = 0;
	buildRangeInfo.transformOffset = 0;

	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = &buildRangeInfo;

	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	CmdBuildAccelerationStructuresKHR(device, commandBuffer, 1, &buildGeometryInfo, &rangeInfo);

	endSingleTimeCommands(commandBuffer);

	vkFreeMemory(device, scratchMemory, nullptr);
	vkDestroyBuffer(device, scratchBuffer, nullptr);
	vkFreeMemory(device, instanceMemory, nullptr);
	vkDestroyBuffer(device, instanceBuffer, nullptr);
}