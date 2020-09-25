#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 Normal;
out vec3 FragPos;
out vec3 vlightPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 lightPos;

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0);
	Normal = mat3(view * model) * aNormal; // Assuming non-uniform scaling in different directions is not done
	FragPos = vec3(view * model * vec4(aPos, 1.0));
	vlightPos = vec3(view * vec4(lightPos, 1.0));
}