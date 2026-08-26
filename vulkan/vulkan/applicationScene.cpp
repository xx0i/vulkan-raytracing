#include "Application.h"

// ============================================================
// scene construction
// ============================================================

void application::simpleDraw()
{
	vertices =
	{
		{{-0.5f, -0.5f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f}, 0.0f, {1.0f, 0.0f}},
		{{ 0.5f, -0.5f, 0.0f}, 0.0f, {0.0f, 1.0f, 0.0f}, 0.0f, {0.0f, 0.0f}},
		{{ 0.5f,  0.5f, 0.0f}, 0.0f, {0.0f, 0.0f, 1.0f}, 0.0f, {0.0f, 1.0f}},
		{{ -0.5f, 0.5f, 0.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f, {1.0f, 1.0f}}
	};

	indices = { 0, 1, 2, 2, 3, 0 };
}

void application::drawShapes()
{
	switch (12)
	{
	case 0: //two simple lambertian spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
		};

		materials =
		{
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
		};

		missShaderColouring = 1;

		break;

	case 1: //two lambertian, two metal spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, 1.0f, -4.5f}, 0.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
		};

		materials =
		{
			{{0.1f, 0.2f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.0f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.8f, 1.0f}, 0.3f, 0.0f, materialType::metal},
			{{0.8f, 0.6f, 0.2f, 1.0f}, 1.0f, 0.0f, materialType::metal}
		};

		missShaderColouring = 1;

		break;

	case 2: //two lambertian, one dieletric (glass), one metal spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, 1.0f, -4.5f}, 0.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
		};

		materials =
		{
			{{0.1f, 0.2f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.0f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.8f, 1.0f}, 0.0f, 1.50f, materialType::dielectric},
			{{0.8f, 0.6f, 0.2f, 1.0f}, 1.0f, 0.0f, materialType::metal}
		};

		missShaderColouring = 1;

		break;

	case 3: //two lambertian, one dieletric (air bubble), one metal spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, 1.0f, -4.5f}, 0.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
		};

		materials =
		{
			{{0.1f, 0.2f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.0f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.8f, 1.0f}, 0.0f, 1.0f / 1.33f, materialType::dielectric},
			{{0.8f, 0.6f, 0.2f, 1.0f}, 1.0f, 0.0f, materialType::metal}
		};

		missShaderColouring = 1;

		break;

	case 4: //two lambertian, one dieletric (hollow glass), one metal spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, -1.0f, -4.5f}, 0.4f, {1.0f, 0.0f, 1.0f, 1.0f}, 0},
			{{0.0f, 1.0f, -4.5f}, 0.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
		};

		materials =
		{
			{{0.1f, 0.2f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.0f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.8f, 1.0f}, 0.0f, 1.50f, materialType::dielectric},
			{{0.8f, 0.8f, 0.8f, 1.0f}, 0.0f, 1.00 / 1.50f, materialType::dielectric},
			{{0.8f, 0.6f, 0.2f, 1.0f}, 1.0f, 0.0f, materialType::metal}
		};

		missShaderColouring = 1;

		break;

	case 5: //random small spheres with three main spheres and a ground sphere
		spheres.push_back({ {0.0f, 0.0f, -1000.0f}, 1000.0f, {0.5f, 0.5f, 0.5f, 1.0f}, 0, {} });
		materials.push_back({ {0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, lambertian, 0 });

		for (int a = -11; a < 11; ++a)
		{
			for (int b = -11; b < 11; ++b)
			{
				float choose_mat = random_float();
				glm::vec3 center = { a + 0.9f * random_float(), b + 0.9f * random_float(), 0.2f };

				if (glm::length(center - glm::vec3(4.0f, 0.0f, 0.2f)) > 0.9f)
				{
					glm::vec4 color;
					float fuzz = 0.0f;
					float ir = 1.5f;
					materialType type;

					if (choose_mat < 0.8f)
					{
						glm::vec3 albedo = random_vec3() * random_vec3();
						color = glm::vec4(albedo, 1.0f);
						type = lambertian;
					}
					else if (choose_mat < 0.95f)
					{
						glm::vec3 albedo = random_vec3(0.5f, 1.0f);
						color = glm::vec4(albedo, 1.0f);
						fuzz = random_float(0.0f, 0.5f);
						type = metal;
					}
					else
					{
						color = glm::vec4(1.0f);
						type = dielectric;
					}

					uint32_t materialIndex = static_cast<uint32_t>(materials.size());
					spheres.push_back({ center, 0.2f, color, materialIndex, {} });
					materials.push_back({ color, fuzz, 1.0f / ir, type, 0 });
				}
			}
		}

		spheres.push_back({ {0.0f, 0.0f, 1.0f}, 1.0f, {1.0f, 1.0f, 1.0f, 1.0f}, static_cast<uint32_t>(materials.size()), {} });
		materials.push_back({ {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 1.5f, dielectric, 0 });

		spheres.push_back({ {-2.0f, 0.0f, 1.0f}, 1.0f, {0.4f, 0.2f, 0.1f, 1.0f}, static_cast<uint32_t>(materials.size()), {} });
		materials.push_back({ {0.4f, 0.2f, 0.1f, 1.0f}, 0.0f, 0.0f, lambertian, 0 });

		spheres.push_back({ {2.0f, 0.0f, 1.0f}, 1.0f, {0.7f, 0.6f, 0.5f, 1.0f}, static_cast<uint32_t>(materials.size()), {} });
		materials.push_back({ {0.7f, 0.6f, 0.5f, 1.0f}, 0.0f, 0.0f, metal, 0 });

		missShaderColouring = 1;

		break;

	case 6: //one textured, one checkered, and one large lambertian spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0, 1},
			{{0.0f, -1.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0, 0, 1},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0},
		};

		materials =
		{
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
		};

		missShaderColouring = 1;

		break;

	case 7: //two perlin noise (marbled) spheres
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0, 0, 0, 1},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0, 0, 0, 1},
		};

		materials =
		{
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
		};

		missShaderColouring = 1;

		break;

	case 8: //5 walls forming an open cube with space between each wall
		quads =
		{
			{ { -1.0f, -0.25f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, 2.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 0.0f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, 2.0f }, 0.0f },
			{ { 0.7677f, -1.75607f, 0.0f }, 0.0f, { 1.4142f, 1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, 2.0f }, 0.0f },
			{ { -1.0f, -0.5f, -0.5f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f },
			{ { -1.0f, -0.5f, 2.5f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f },
		};

		materials =
		{
			{{1.0f, 0.2f, 0.2f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.2f, 1.0f, 0.2f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.2f, 0.2f, 1.0f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.2f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.8f, 0.8f, 0.2f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
		};

		missShaderColouring = 1;

		break;

	case 9: //two perlin noise (marbled) spheres with one quad light
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0, 0, 0, 1},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0, 0, 0, 1},
		};

		quads =
		{
			{ { 1.5f, -0.55f, -5.0f }, 0.0f, { -0.7071f,  -0.7071f, 0.0f }, 0.0f, { 0.0f, 0.0f, 1.0f }, 0.0f },
		};

		materials =
		{
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, materialType::diffuseLight, 0, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f}
		};

		missShaderColouring = 0;

		break;

	case 10: //two perlin noise (marbled) spheres with one quad light and one sphere light
		spheres =
		{
			{{0.0f, 0.0f, -4.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0, 0, 0, 1},
			{{0.0f, -1.0f, -100.5f}, 95.5f, {0.0f, 1.0f, 0.0f, 1.0f}, 0, 0, 0, 1},
			{{0.0f, 0.0f, -2.5f}, 0.5f, {1.0f, 0.0f, 1.0f, 1.0f}, 0, 0, 0, 0},
		};

		quads =
		{
			{ { 1.5f, -0.55f, -5.0f }, 0.0f, { -0.7071f,  -0.7071f, 0.0f }, 0.0f, { 0.0f, 0.0f, 1.0f }, 0.0f },
		};

		materials =
		{
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.5f, 0.5f, 0.5f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, materialType::diffuseLight, 0, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f},
			{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, materialType::diffuseLight, 0, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f}
		};

		missShaderColouring = 0;

		break;

	case 11: //empty cornell box
		quads =
		{
			{ { -1.3535f, -0.8535f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, 2.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 2.0f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, -2.0f }, 0.0f },
			{ { 0.0607f, -2.2677f, 2.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, -2.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 0.0f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 2.0f }, 0.0f, {  1.4142f,  1.4142f, 0.0f }, 0.0f, {  1.4142f, -1.4142f, 0.0f }, 0.0f },
			{ { 0.0f, -0.35f, 1.9999f }, 0.0f,{ 0.5657f, -0.5657f, 0.0f }, 0.0f, { -0.5657f,  -0.5657f, 0.0f }, 0.0f },
		};

		materials =
		{
			{{0.12f, 0.45f, 0.15f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.73f, 0.73f, 0.73f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.65f, 0.05f, 0.05f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.73f, 0.73f, 0.73f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.73f, 0.73f, 0.73f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, materialType::diffuseLight, 0, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f},
		};

		missShaderColouring = 0;

		break;

	case 12: //classic cornell box
		quads =
		{
			{ { -1.3535f, -0.8535f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, 2.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 2.0f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, -2.0f }, 0.0f },
			{ { 0.0607f, -2.2677f, 2.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f, { 0.0f, 0.0f, -2.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 0.0f }, 0.0f, { 1.4142f, -1.4142f, 0.0f }, 0.0f, { 1.4142f,  1.4142f, 0.0f }, 0.0f },
			{ { -1.3535f, -0.8535f, 2.0f }, 0.0f, {  1.4142f,  1.4142f, 0.0f }, 0.0f, {  1.4142f, -1.4142f, 0.0f }, 0.0f },
			{ { 0.0f, -0.35f, 1.9999f }, 0.0f,{ 0.5657f, -0.5657f, 0.0f }, 0.0f, { -0.5657f,  -0.5657f, 0.0f }, 0.0f },
		};

		makeRotatedBox({ -0.6f, -1.0f, 0.0f }, { 0.1f, -0.3f, 1.5f }, glm::radians(15.0f));
		makeBox({ 0.4f, -1.2f, 0.0f }, { 1.0f, -0.5f, 0.8f });

		materials =
		{
			{{0.12f, 0.45f, 0.15f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.73f, 0.73f, 0.73f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.65f, 0.05f, 0.05f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.73f, 0.73f, 0.73f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{0.73f, 0.73f, 0.73f, 1.0f}, 0.0f, 0.0f, materialType::lambertian},
			{{1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, 0.0f, materialType::diffuseLight, 0, {1.0f, 1.0f, 1.0f, 4.0f}, 0.0f},
		};

		{
			size_t requiredMaterials = spheres.size() + quads.size();
			while (materials.size() < requiredMaterials)
			{
				materials.push_back({ { 0.73f, 0.73f, 0.73f, 1.0f }, 0.0f, 0.0f, materialType::lambertian });
			}
		}

		missShaderColouring = 0;

		break;


	default:
		break;
	}

	for (uint32_t i = 0; i < spheres.size(); i++)
	{
		VkAabbPositionsKHR aabb =
		{
			spheres[i].center.x - spheres[i].radius,
			spheres[i].center.y - spheres[i].radius,
			spheres[i].center.z - spheres[i].radius,
			spheres[i].center.x + spheres[i].radius,
			spheres[i].center.y + spheres[i].radius,
			spheres[i].center.z + spheres[i].radius
		};

		aabbs.push_back(aabb);
		geoTypes.push_back(geometryType::sphereShape);

		aabbObject obj{};
		obj.type = geometryType::sphereShape;
		obj.geoIndex = i;
		obj.matIndex = i;
		obj.aabb = aabb;
		aabbObjects.push_back(obj);
	}

	for (uint32_t i = 0; i < quads.size(); i++)
	{
		glm::vec3 p0 = quads[i].origin;
		glm::vec3 p1 = quads[i].origin + quads[i].edgeU;
		glm::vec3 p2 = quads[i].origin + quads[i].edgeV;
		glm::vec3 p3 = quads[i].origin + quads[i].edgeU + quads[i].edgeV;

		glm::vec3 minCorner = glm::min(glm::min(p0, p1), glm::min(p2, p3));
		glm::vec3 maxCorner = glm::max(glm::max(p0, p1), glm::max(p2, p3));

		float pad = 0.001f;

		VkAabbPositionsKHR aabb =
		{
			minCorner.x - pad,
			minCorner.y - pad,
			minCorner.z - pad,
			maxCorner.x + pad,
			maxCorner.y + pad,
			maxCorner.z + pad
		};

		aabbs.push_back(aabb);
		geoTypes.push_back(geometryType::quadShape);

		aabbObject obj{};
		obj.type = geometryType::quadShape;
		obj.geoIndex = i;
		obj.matIndex = i + spheres.size();
		obj.aabb = aabb;
		aabbObjects.push_back(obj);
	}
}

void application::makeBox(glm::vec3 p0, glm::vec3 p1)
{
	quads.push_back({ {p0.x, p0.y, p1.z}, 0.0f,
				  {p1.x - p0.x, 0.0f, 0.0f}, 0.0f,
				  {0.0f, p1.y - p0.y, 0.0f}, 0.0f });

	quads.push_back({ {p1.x, p0.y, p0.z}, 0.0f,
				  {p0.x - p1.x, 0.0f, 0.0f}, 0.0f,
				  {0.0f, p1.y - p0.y, 0.0f}, 0.0f });

	quads.push_back({ {p0.x, p0.y, p0.z}, 0.0f,
				  {0.0f, 0.0f, p1.z - p0.z}, 0.0f,
				  {0.0f, p1.y - p0.y, 0.0f}, 0.0f });

	quads.push_back({ {p1.x, p0.y, p1.z}, 0.0f,
				  {0.0f, 0.0f, p0.z - p1.z}, 0.0f,
				  {0.0f, p1.y - p0.y, 0.0f}, 0.0f });

	quads.push_back({ {p0.x, p0.y, p0.z}, 0.0f,
				  {p1.x - p0.x, 0.0f, 0.0f}, 0.0f,
				  {0.0f, 0.0f, p1.z - p0.z}, 0.0f });

	quads.push_back({ {p0.x, p1.y, p1.z}, 0.0f,
				  {p1.x - p0.x, 0.0f, 0.0f}, 0.0f,
				  {0.0f, 0.0f, p0.z - p1.z}, 0.0f });

}

void application::makeRotatedBox(const glm::vec3& pMin, const glm::vec3& pMax, float angle)
{
	glm::vec3 corners[8] = {
		{ pMin.x, pMin.y, pMin.z },
		{ pMax.x, pMin.y, pMin.z },
		{ pMax.x, pMax.y, pMin.z },
		{ pMin.x, pMax.y, pMin.z },

		{ pMin.x, pMin.y, pMax.z },
		{ pMax.x, pMin.y, pMax.z },
		{ pMax.x, pMax.y, pMax.z },
		{ pMin.x, pMax.y, pMax.z },
	};

	glm::vec3 center = (pMin + pMax) * 0.5f;

	float c = cos(angle);
	float s = sin(angle);

	auto rotateZ = [&](const glm::vec3& p) -> glm::vec3 {
		glm::vec3 q = p - center;
		return {
			c * q.x - s * q.y + center.x,
			s * q.x + c * q.y + center.y,
			q.z + center.z
		};
		};

	for (int i = 0; i < 8; i++) {
		corners[i] = rotateZ(corners[i]);
	}

	quads.push_back({ corners[0], 0.0f, corners[1] - corners[0], 0.0f, corners[3] - corners[0], 0.0f });
	quads.push_back({ corners[4], 0.0f, corners[5] - corners[4], 0.0f, corners[7] - corners[4], 0.0f });
	quads.push_back({ corners[0], 0.0f, corners[1] - corners[0], 0.0f, corners[4] - corners[0], 0.0f });
	quads.push_back({ corners[2], 0.0f, corners[3] - corners[2], 0.0f, corners[6] - corners[2], 0.0f });
	quads.push_back({ corners[0], 0.0f, corners[3] - corners[0], 0.0f, corners[4] - corners[0], 0.0f });
	quads.push_back({ corners[1], 0.0f, corners[2] - corners[1], 0.0f, corners[5] - corners[1], 0.0f });
}

float application::random_float(float min, float max)
{
	static thread_local std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<float> distribution(min, max);
	return distribution(generator);
}

glm::vec3 application::random_vec3(float min, float max)
{
	return glm::vec3(
		random_float(min, max),
		random_float(min, max),
		random_float(min, max)
	);
}

void application::loadModel()
{
	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warning, error;

	if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, modelPath.c_str()))
	{
		throw std::runtime_error(warning + error);
	}

	std::unordered_map<vertex, uint32_t> uniqueVertices{};

	for (const auto& shape : shapes)
	{
		for (const auto& index : shape.mesh.indices)
		{
			vertex vert{};

			vert.pos =
			{
				attributes.vertices[3 * index.vertex_index + 0],
				attributes.vertices[3 * index.vertex_index + 1],
				attributes.vertices[3 * index.vertex_index + 2]
			};

			vert._pad0 = 0.0f;

			vert.texCoord =
			{
				attributes.texcoords[2 * index.texcoord_index + 0],
				1.0f - attributes.texcoords[2 * index.texcoord_index + 1]
			};

			vert._pad1 = 0.0f;

			vert.colour = { 1.0f, 1.0f, 1.0f };

			if (uniqueVertices.count(vert) == 0)
			{
				uniqueVertices[vert] = static_cast<uint32_t>(vertices.size());
				vertices.push_back(vert);
			}

			vert._pad2 = { 0.0f, 0.0f };

			indices.push_back(uniqueVertices[vert]);
		}
	}
}