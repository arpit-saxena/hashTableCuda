#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>

#include "Mesh.h"

Mesh::Mesh(const std::string fileName) {
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(fileName.c_str(), aiProcess_Triangulate);

	if (!(scene && scene->HasMeshes())) {
		std::cerr << "Mesh load failed!: " << fileName << std::endl
				  << importer.GetErrorString() << std::endl;
		exit(1);
	}

	if (scene->mNumMeshes == 0) {
		std::cerr << fileName << " has no meshes" << std::endl;
	}

	const aiMesh* model = scene->mMeshes[0];
	numVertices = model->mNumVertices;
	vertices = new float4[numVertices];
	for (int i = 0; i < model->mVertices; i++) {
		const aiVector3D pos = model->mVertices[i];
		vertices[i] = make_float4(pos.x, pos.y, pos.z, 1.0f);
	}
}

Mesh::~Mesh() {
	delete[] vertices;
}

int Mesh::getNumVertices() {
	return numVertices;
}