#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS // stupid crap VS forcing me to use the safer sprintf_s instead of sprintf
#endif

#include "render.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "kernel.cuh"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "shader_s.h"

namespace {
	float lightcubevertices[] = {
		// vertices
		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,

		-0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,

		-0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f, -0.5f,
		-0.5f, -0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,

		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,

		-0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f, -0.5f,
		 0.5f, -0.5f,  0.5f,
		 0.5f, -0.5f,  0.5f,
		-0.5f, -0.5f,  0.5f,
		-0.5f, -0.5f, -0.5f,

		-0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f, -0.5f,
		 0.5f,  0.5f,  0.5f,
		 0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f,  0.5f,
		-0.5f,  0.5f, -0.5f
	};

	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
	}

	void checkShaderCompileStatus(unsigned int shader, std::string typeOfShader) {
		int  success;
		char infoLog[512];
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER" <<typeOfShader << "::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
	}

	void checkProgramLinkStatus(unsigned int shaderProgram) {
		int  success;
		char infoLog[512];
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}
	}

	const glm::vec3 lightPos = glm::vec3(1.2f, 1.0f, 2.0f);

	void prepdrawlightingCube(GLuint& lightCubeVAO, GLuint& lightCubeVBO) {
		glGenVertexArrays(1, &lightCubeVAO);
		glBindVertexArray(lightCubeVAO);

		glGenBuffers(1, &lightCubeVBO);
		glBindBuffer(GL_ARRAY_BUFFER, lightCubeVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(lightcubevertices), lightcubevertices, GL_STATIC_DRAW);
		// set the vertex attribute 
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void drawlightingCube(GLuint lightCubeVAO, Shader* lightCubeShader, glm::mat4 view) {
		auto model = glm::mat4(1.0f);
		model = glm::translate(model, lightPos);
		model = glm::scale(model, glm::vec3(0.2f));

		lightCubeShader->use();
		lightCubeShader->setMatrix4fv("model", model);
		lightCubeShader->setMatrix4fv("view", view);
		glBindVertexArray(lightCubeVAO);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

	double mouse_old_x, mouse_old_y;
	int mouse_buttons = 0;
	double rotate_x = 0.0, rotate_y = 0.0;
	double translate_z = -3.0;

	void motion(GLFWwindow* window)
	{
		double x, y;
		glfwGetCursorPos(window, &x, &y);
		double dx, dy;
		dx = x - mouse_old_x;
		dy = y - mouse_old_y;

		if (mouse_buttons & (1<<GLFW_MOUSE_BUTTON_LEFT))
		{
			rotate_x += dy * 0.2;
			rotate_y += dx * 0.2;
		}
		else if (mouse_buttons & (1 << GLFW_MOUSE_BUTTON_RIGHT))
		{
			translate_z += dy * 0.01f;
		}
		mouse_old_x = x;
		mouse_old_y = y;
	}

	void mouse(GLFWwindow* window, int button, int action, int mods)
	{
		if (action == GLFW_PRESS)
		{
			mouse_buttons |= 1 << button;
		}
		else if (action == GLFW_RELEASE)
		{
			mouse_buttons = 0;
		}
	}

	void prepdraw(unsigned int &VAO, unsigned int &VBO) {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		const unsigned size = CUDA::numOfTriangles * 18;
		glBufferData(GL_ARRAY_BUFFER, size*sizeof(float), 0, GL_DYNAMIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// color attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void draw(GLuint VAO, GLuint lightCubeVAO, Shader* sh, Shader* lightCubeShader) {
		glm::mat4 model = glm::mat4(1.0f);

		glm::mat4 view = glm::mat4(1.0f);
		view = glm::translate(view, glm::vec3(0.0, 0.0, translate_z));
		view = glm::rotate(view, (float)glm::radians(rotate_x), glm::vec3(1.0, 0.0, 0.0));
		view = glm::rotate(view, (float)glm::radians(rotate_y), glm::vec3(0.0, 1.0, 0.0));    

		sh->use();
		sh->setMatrix4fv("model", model);
		sh->setMatrix4fv("view", view);

		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 36);

		drawlightingCube(lightCubeVAO, lightCubeShader, view);
	}
}

int OpenGL::render() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glViewport(0, 0, 800, 600);
	
	glEnable(GL_DEPTH_TEST);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetMouseButtonCallback(window, mouse);

	Shader sh("render/Shaders/shader.vs", "render/Shaders/shader.fs");
	sh.use();
	sh.set3fv("objectColor", 1.0f, 0.5f, 0.31f);
	sh.set3fv("lightColor", 1.0f, 1.0f, 1.0f);
	sh.set3fv("lightPos", lightPos.x, lightPos.y, lightPos.z);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
	sh.setMatrix4fv("projection", projection);
	Shader lightCubeShader("render/Shaders/lightCubeShader.vs", "render/Shaders/lightCubeShader.fs");
	lightCubeShader.use();
	lightCubeShader.setMatrix4fv("projection", projection);

	unsigned int VAO, VBO, lightCubeVAO, lightCubeVBO;
	prepdraw(VAO, VBO);
	struct cudaGraphicsResource* vbo_resource;
	CUDA::registerVBO(VBO, &vbo_resource);
	prepdrawlightingCube(lightCubeVAO, lightCubeVBO);

	double lastTime = glfwGetTime();
	const double step = 0.1;
	int nbFrames = 0;

	while (!glfwWindowShouldClose(window))
	{
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= step) { // After atleast 1 sec has elapsed
			char titlestr[30];
			sprintf(titlestr, "LearnOpenGL | FPS=%.2f", (double)nbFrames/(currentTime - lastTime));
			glfwSetWindowTitle(window, titlestr);
			nbFrames = 0;
			lastTime = currentTime;
		}

		CUDA::runCuda(&vbo_resource, currentTime);

		processInput(window);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		motion(window);
		draw(VAO,lightCubeVAO, &sh, &lightCubeShader);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteVertexArrays(1, &lightCubeVAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &lightCubeVBO);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}