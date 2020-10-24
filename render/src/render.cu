#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS // stupid crap VS forcing me to use the safer sprintf_s instead of sprintf
#endif

#include "render.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#include "shader_s.h"

glm::mat4 CUDA::trans_mats[2];

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
	const glm::vec3 lightPos = glm::vec3(1.2f, 1.0f, 2.0f);

	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
	}

	float getCurrAspectRatio() {
		GLint data[4];
		glGetIntegerv(GL_VIEWPORT, data);
		return (float)data[2] / (float)data[3];
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
	
	glm::mat4 makeViewMat() {
		glm::mat4 view = glm::mat4(1.0f);
		view = glm::translate(view, glm::vec3(0.0, 0.0, translate_z));
		view = glm::rotate(view, (float)glm::radians(rotate_x), glm::vec3(1.0, 0.0, 0.0));
		view = glm::rotate(view, (float)glm::radians(rotate_y), glm::vec3(0.0, 1.0, 0.0));
		return view;
	}

	void setViewMat(glm::mat4 view, Shader* s) {
		s->use();
		s->setMatrix4fv("view", view);
	}

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

	void drawlightingCube(GLuint lightCubeVAO, Shader* lightCubeShader) {
		auto model = glm::mat4(1.0f);
		model = glm::translate(model, lightPos);
		model = glm::scale(model, glm::vec3(0.2f));

		lightCubeShader->use();
		lightCubeShader->setMatrix4fv("model", model);
		glBindVertexArray(lightCubeVAO);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}

}

__host__ void CUDA::launch_kernel(Triangle* buffer[2], unsigned numTriangles[2], Mesh meshes[2], HashTable * d_h, glm::mat4 transformation_mat[2])
{
	unsigned totalTriangles = numTriangles[0] + numTriangles[1];
	int threadsPerBlock = THREADS_PER_BLOCK, numBlocks = CEILDIV(totalTriangles, threadsPerBlock);
	CUDA::triangleKernel <<< numBlocks, threadsPerBlock >>> (buffer[0], buffer[1], numTriangles[0], numTriangles[1],
		meshes[0].triangles, meshes[1].triangles, d_h, transformation_mat);
	gpuErrchk( cudaDeviceSynchronize() );
}

__device__ void CUDA::transform(Triangle* t, const glm::mat4 transformation_mat)
{
	for (int i = 0; i < 3; ++i) {
		Vertex * v = t->vertices + i;
		glm::vec4 pt = glm::vec4(v->point[0], v->point[1], v->point[2], 1.0f);
		glm::vec4 nm = glm::vec4(v->normal[0], v->normal[1], v->normal[2], 1.0f);
		pt = transformation_mat * pt;
		nm = transformation_mat * nm;
		v->point[0] = pt.x;		v->normal[0] = nm.x;
		v->point[1] = pt.y;		v->normal[1] = nm.y;
		v->point[2] = pt.z;		v->normal[2] = nm.z;
	}
}

__global__ void CUDA::triangleKernel(Triangle* buffer0, Triangle* buffer1, unsigned numTriangles0, unsigned numTriangles1,
	Triangle * mesh0, Triangle * mesh1, HashTable* d_h, glm::mat4 * transformation_mat)
{
	const unsigned totalTriangles = numTriangles0 + numTriangles1;
	const unsigned maxThreadsToRun = CEILDIV(totalTriangles, warpSize) * warpSize;
	Triangle* buffer = nullptr;
	Triangle* mesh = nullptr;
	int meshindex = -1;

	int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	while (global_thread_id < maxThreadsToRun) {
		unsigned triangleindex;
		if (global_thread_id < numTriangles0) {
			buffer = buffer0;
			mesh = mesh0;
			meshindex = 0;
			triangleindex = global_thread_id;
		}
		else if(global_thread_id < totalTriangles) {
			buffer = buffer1;
			mesh = mesh1;
			meshindex = 1;
			triangleindex = global_thread_id - numTriangles0;
		}
		else {		// This is an invalid thread, all the triangles have already been assigned to previous threads
					// This thread just needs to run for the warp-cooperative HashTable functions
			buffer = mesh = nullptr;
			triangleindex = 0;
			meshindex = 2;
		}
		__syncwarp();
		assert(__activemask() == WARP_MASK);
		Triangle* currtriangle = nullptr;
		if (meshindex <= 1) {
			currtriangle = mesh + triangleindex;
		}
		CUDA::updateTrianglePosition(currtriangle, triangleindex, meshindex, d_h, transformation_mat[meshindex]);
		if (meshindex <= 1) {
			buffer[triangleindex] = *currtriangle;
		}

		global_thread_id += gridDim.x * blockDim.x;
	}
}

void OpenGLScene::runCuda()
{
	// map OpenGL buffer object for writing from CUDA
	Triangle* dptr[2];
	gpuErrchk(cudaGraphicsMapResources(2, vbo_resource));

	size_t num_bytes[2];
	unsigned numTriangles[2];
	for (int i = 0; i < 2; ++i) {
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)(dptr + i), num_bytes + i, vbo_resource[i]));
		if (num_bytes[i] != this->meshes[i].numTriangles * sizeof(Triangle)) {
			std::cerr << "VBO " << i << " not initialized properly, no. of bytes in VBO found after mapping to CUDA = "
				<< num_bytes[i] << " and no. of bytes of triangles in mesh = " << this->meshes[i].numTriangles * sizeof(Triangle)
				<< std::endl;
			assert(num_bytes[i] == this->meshes[i].numTriangles * sizeof(Triangle));
			return;
		}
		else {
			numTriangles[i] = num_bytes[i] / sizeof(Triangle);
		}
	}

	auto model_mat = this->trans_model_mat;
	if (this->isFirstFrame) {
		model_mat = this->init_model_mat;
	}
	else if (this->collided) {
		model_mat = this->identity_model_mat;
	}

	CUDA::preprocess();
	CUDA::launch_kernel(dptr, numTriangles, this->meshes, this->d_h, model_mat);

	if (!this->collided) {
		collisionMarker::init(this->meshes);
		this->collided = CUDA::detectCollision(this->meshes, this->d_h);
	}

	// unmap buffer object
	gpuErrchk(cudaGraphicsUnmapResources(2, vbo_resource));
	this->isFirstFrame = false;
}

OpenGLScene::OpenGLScene(Mesh h_meshes[2], glm::mat4 init_model_mat[2], glm::mat4 trans_model_mat[2], HashTable * d_h) {
	for (int i = 0; i < 2; ++i) {
		for (int t = 0; t < h_meshes[i].numTriangles; ++t) {
			for (int v = 0; v < 3; ++v) {
				h_meshes[i].triangles[t].vertices[v].hasCollided = 0.0f;	//All meshes will be uncollided initially
			}
		}
		meshes[i].numTriangles = h_meshes[i].numTriangles;
		gpuErrchk(cudaMalloc((void**)&meshes[i].triangles, meshes[i].numTriangles * sizeof(Triangle)));
		gpuErrchk(cudaMemcpy(meshes[i].triangles, h_meshes[i].triangles, meshes[i].numTriangles * sizeof(Triangle), cudaMemcpyDefault));
	}

	glm::mat4 identity_model_mat[2] = { glm::mat4(1.0f), glm::mat4(1.0f) };

	gpuErrchk(cudaMalloc((void**)&(this->init_model_mat), 2 * sizeof(glm::mat4)));
	gpuErrchk(cudaMalloc((void**)&(this->trans_model_mat), 2 * sizeof(glm::mat4)));
	gpuErrchk(cudaMalloc((void**)&(this->identity_model_mat), 2 * sizeof(glm::mat4)));
	gpuErrchk(cudaMemcpy(this->init_model_mat, init_model_mat, 2 * sizeof(glm::mat4), cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(this->trans_model_mat, trans_model_mat, 2 * sizeof(glm::mat4), cudaMemcpyDefault));
	gpuErrchk(cudaMemcpy(this->identity_model_mat, identity_model_mat, 2 * sizeof(glm::mat4), cudaMemcpyDefault));

	CUDA::trans_mats[0] = trans_model_mat[0];
	CUDA::trans_mats[1] = trans_model_mat[1];

	this->d_h = d_h;
	this->isFirstFrame = true;
	this->collided = false;
}

namespace {
	template<class T>
	void safecudafree(T** ptr) {
		if (*ptr != nullptr) {
			gpuErrchk(cudaFree(*ptr));
			*ptr = nullptr;
		}
	}
}

OpenGLScene::~OpenGLScene() {
	for (int i = 0; i < 2; ++i) {
		::safecudafree(&(meshes[i].triangles));
	}
	::safecudafree(&init_model_mat);
	::safecudafree(&trans_model_mat);
}

void OpenGLScene::registerVBO()
{
	for (int i = 0; i < 2; ++i) {
		gpuErrchk(cudaGraphicsGLRegisterBuffer(vbo_resource+i, VBO[i], cudaGraphicsRegisterFlagsWriteDiscard));
	}
}

void OpenGLScene::prepdraw()
{
	glGenVertexArrays(2, VAO);
	glGenBuffers(2, VBO);

	for (int i = 0; i < 2; ++i) {
		glBindVertexArray(VAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);
		const size_t size = this->meshes[i].numTriangles * sizeof(Triangle);
		glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, point));
		glEnableVertexAttribArray(0);
		// normal attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
		glEnableVertexAttribArray(1);
		// collided attribute
		glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, hasCollided));
		glEnableVertexAttribArray(2);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void OpenGLScene::draw()
{
	glm::mat4 model = glm::mat4(1.0f);

	sh->use();
	sh->setMatrix4fv("model", model);

	for (int i = 0; i < 2; ++i) {
		glBindVertexArray(VAO[i]);
		glDrawArrays(GL_TRIANGLES, 0, this->meshes[i].numTriangles*3);
	}
	glBindVertexArray(0);
}

void OpenGLScene::destroyGLobjs()
{
	gpuErrchk(cudaGraphicsUnregisterResource(this->vbo_resource[0]));
	gpuErrchk(cudaGraphicsUnregisterResource(this->vbo_resource[1]));
	
	glDeleteVertexArrays(2, this->VAO);
	glDeleteBuffers(2, this->VBO);
}

int OpenGLScene::render() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow* window = glfwCreateWindow(800, 600, "hashTableCuda", NULL, NULL);
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
	sh.set3fv("objectColor", 1.0f, 0.5f, 0.31f);		// Coral
	sh.set3fv("collidedColor", 0.698f, 0.133f, 0.133f);	// FireBrick
	sh.set3fv("lightColor", 1.0f, 1.0f, 1.0f);
	sh.set3fv("lightPos", lightPos.x, lightPos.y, lightPos.z);

	glm::mat4 projection = glm::perspective(glm::radians(45.0f), ::getCurrAspectRatio(), 0.1f, 100.0f);
	sh.setMatrix4fv("projection", projection);

	this->sh = &sh;

	Shader lightCubeShader("render/Shaders/lightCubeShader.vs", "render/Shaders/lightCubeShader.fs");
	lightCubeShader.use();
	lightCubeShader.setMatrix4fv("projection", projection);

	unsigned int lightCubeVAO, lightCubeVBO;
	::prepdrawlightingCube(lightCubeVAO, lightCubeVBO);

	this->prepdraw();
	this->registerVBO();

	double lastTime = glfwGetTime();
	const double step = 0.1;
	int nbFrames = 0;

	CUDA::initCollisionDet(this->meshes);

	while (!glfwWindowShouldClose(window))
	{
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= step) { // After atleast 1 sec has elapsed
			char titlestr[30];
			sprintf(titlestr, "hashTableCuda | FPS=%.2f", (double)nbFrames/(currentTime - lastTime));
			glfwSetWindowTitle(window, titlestr);
			nbFrames = 0;
			lastTime = currentTime;
		}

		this->runCuda();

		::processInput(window);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		::motion(window);
		auto view = ::makeViewMat();
		::setViewMat(view, &sh);
		::setViewMat(view, &lightCubeShader);
		drawlightingCube(lightCubeVAO, &lightCubeShader);
		this->draw();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glDeleteVertexArrays(1, &lightCubeVAO);
	glDeleteBuffers(1, &lightCubeVBO);
	this->destroyGLobjs();

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}