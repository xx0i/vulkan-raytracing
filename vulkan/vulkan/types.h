#pragma once
#include "common.h"

const uint32_t width = 800;
const uint32_t height = 600;

const std::string modelPath = "models/viking_room.obj";
const std::string texturePath = "textures/earthmap.jpg";

const int maxFramesInFlight = 3;

const std::vector<const char*> validationLayers =
{
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
	VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
	VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
	VK_KHR_SPIRV_1_4_EXTENSION_NAME,
	VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
	VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
	VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
	VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct queueFamilyIndices
{
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete()
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct swapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct vertex
{
	glm::vec3 pos;
	float _pad0 = 0.0f;
	glm::vec3 colour;
	float _pad1 = 0.0f;
	glm::vec2 texCoord;
	glm::vec2 _pad2 = glm::vec2(0.0f);

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(vertex, colour);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(vertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const vertex& other) const
	{
		return pos == other.pos && colour == other.colour && texCoord == other.texCoord;
	}
};

namespace std
{
	template<> struct hash<vertex>
	{
		size_t operator()(vertex const& vert) const
		{
			return ((hash<glm::vec3>()(vert.pos) ^
				(hash<glm::vec3>()(vert.colour) << 1)) >> 1) ^
				(hash<glm::vec2>()(vert.texCoord) << 1);
		}
	};
}

struct alignas(16) uniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 prevView;
	glm::mat4 prevProj;
	glm::mat4 projUnjittered;
	glm::mat4 prevProjUnjittered;
};

struct camera
{
	glm::vec3 position;
	float yaw;
	float pitch;

	float speed;
	float sensitivity;

	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;
};

enum geometryType : uint32_t
{
	sphereShape = 0,
	quadShape = 1
};

struct sphere
{
	glm::vec3 center;
	float radius;
	glm::vec4 colour;
	uint32_t normalColouring;
	uint32_t textured;
	uint32_t checkered;
	uint32_t perlinNoise;
};

// Size: 48 bytes total
struct quad
{
	glm::vec3 origin; // 12 bytes
	float pad0;       // 4 bytes  (Offset 16)
	glm::vec3 edgeU;  // 12 bytes
	float pad1;       // 4 bytes  (Offset 32)
	glm::vec3 edgeV;  // 12 bytes
	float pad2;       // 4 bytes  (Offset 48)
};

struct aabbObject
{
	geometryType type;
	uint32_t geoIndex;
	uint32_t matIndex;
	VkAabbPositionsKHR aabb;
	VkAccelerationStructureKHR blas;
	VkBuffer blasBuffer = VK_NULL_HANDLE;
	VkDeviceMemory blasMemory = VK_NULL_HANDLE;
	VkDeviceAddress blasDeviceAddress = 0;
};

struct AabbObjectGPU
{
	uint32_t type;
	uint32_t geoIndex;
	uint32_t matIndex;
	uint32_t pad0;
};

enum materialType : uint32_t
{
	lambertian = 0,
	metal = 1,
	dielectric = 2,
	isotropic = 3,
	diffuseLight = 4
};

struct alignas(16) material
{
	glm::vec4 albedo;
	float fuzz;
	float refractionIndex;
	materialType matType;
	uint32_t padding;
	glm::vec4 emission;
	float padding2;
};

struct indexUniformBufferObject
{
	uint32_t imageIndex;
};

struct pushConstants
{
	uint32_t frameIndex;
	uint32_t missColour;
	uint32_t stepSize;
	uint32_t isFinalPass;
	uint32_t isTemporalPass;
};