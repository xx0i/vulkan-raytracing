#include "Application.h"

// ============================================================
// GLFW callbacks
// ============================================================

void application::frameBufferResizeCallback(GLFWwindow* window, int width, int height)
{
	auto app = reinterpret_cast<application*>(glfwGetWindowUserPointer(window));
	app->frameBufferResized = true;
}

void application::mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	auto app = reinterpret_cast<application*>(glfwGetWindowUserPointer(window));
	app->handleMouseCallback(xpos, ypos);
}

void application::handleMouseCallback(double xpos, double ypos)
{
	if (!isMouseCaptured)
	{
		// Reset firstMouse flag so the camera doesn't jump when re-entering captured mode
		firstMouse = true;
		return;
	}

	if (firstMouse)
	{
		lastMouseX = xpos;
		lastMouseY = ypos;
		firstMouse = false;
		return; // Return immediately on initial focus!
	}

	// Accumulate movement deltas across GLFW callback events
	accumulatedDeltaX += static_cast<float>(xpos - lastMouseX);
	accumulatedDeltaY += static_cast<float>(lastMouseY - ypos);

	lastMouseX = xpos;
	lastMouseY = ypos;
}

// ============================================================
// camera vectors
// ============================================================

void application::updateCameraVectors()
{
	glm::vec3 front;
	front.x = cos(camera.pitch) * cos(camera.yaw);
	front.y = cos(camera.pitch) * sin(camera.yaw);
	front.z = sin(camera.pitch);
	camera.front = glm::normalize(front);

	camera.right = glm::normalize(glm::cross(camera.front, glm::vec3(0.0f, 0.0f, 1.0f)));
	camera.up = glm::normalize(glm::cross(camera.right, camera.front));
}

// ============================================================
// input processing
// ============================================================

void application::processMouse(float deltaX, float deltaY)
{
	constexpr float MAX_DELTA = 50.0f;
	deltaX = glm::clamp(deltaX, -MAX_DELTA, MAX_DELTA);
	deltaY = glm::clamp(deltaY, -MAX_DELTA, MAX_DELTA);

	if (std::abs(deltaX) < 1e-4f && std::abs(deltaY) < 1e-4f)
	{
		return;
	}

	camera.yaw -= deltaX * camera.sensitivity;
	camera.pitch += deltaY * camera.sensitivity;

	camera.pitch = glm::clamp(camera.pitch, glm::radians(-89.0f), glm::radians(89.0f));

	updateCameraVectors();
}

bool application::processKeyboard(float deltaTime)
{
	// --- MOUSE CAPTURE TOGGLE (TAB) ---
	static bool tabKeyPressed = false;
	if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS)
	{
		if (!tabKeyPressed)
		{
			isMouseCaptured = !isMouseCaptured;

			if (isMouseCaptured)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
			tabKeyPressed = true;
		}
	}
	else if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_RELEASE)
	{
		tabKeyPressed = false;
	}

	// --- CAMERA MOVEMENT ---
	// Early exit if mouse is unlocked for UI interactions
	if (!isMouseCaptured)
	{
		return false;
	}

	float velocity = camera.speed * deltaTime;
	glm::vec3 originalPosition = camera.position;

	if (forward)
	{
		camera.position += camera.front * velocity;
	}
	if (backward)
	{
		camera.position -= camera.front * velocity;
	}
	if (left)
	{
		camera.position -= camera.right * velocity;
	}
	if (right)
	{
		camera.position += camera.right * velocity;
	}
	if (up)
	{
		camera.position += camera.up * velocity;
	}
	if (down)
	{
		camera.position -= camera.up * velocity;
	}
	return originalPosition != camera.position;
}
