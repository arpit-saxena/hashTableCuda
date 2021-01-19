#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <cassert>
#include <iostream>

#include "Importer.cuh"

Mesh import(std::string fileName) {
  Assimp::Importer importer;

  importer.SetPropertyInteger(
      AI_CONFIG_PP_SBP_REMOVE,
      aiPrimitiveType_LINE |
          aiPrimitiveType_POINT);  // Setting this to remove points and lines =>
                                   // Have only triangles

  const aiScene *scene =
      importer.ReadFile(fileName, aiProcess_Triangulate | aiProcess_GenNormals |
                                      aiProcess_SortByPType);

  Mesh mesh;

  if (!scene) {
    std::cerr << "Unable to import " << fileName << std::endl;
    std::cerr << importer.GetErrorString() << std::endl;
    return mesh;
  }

  if (scene->mNumMeshes < 1) {
    std::cerr << "Scene contains no meshes" << std::endl;
    return mesh;
  }

  aiMesh *assimpMesh = scene->mMeshes[0];

  if (!assimpMesh->HasFaces()) {
    std::cerr << "Mesh contains no faces. Can't proceed" << std::endl;
    return mesh;
  } else if (!assimpMesh->HasNormals()) {
    std::cerr
        << "Mesh contains no normals, and we like lighting!! Can't proceed"
        << std::endl;
    return mesh;
  }

  aiVector3D *vertices = assimpMesh->mVertices, *normals = assimpMesh->mNormals;
  mesh.numTriangles = assimpMesh->mNumFaces;
  mesh.triangles = new Triangle[mesh.numTriangles];
  for (int i = 0; i < assimpMesh->mNumFaces; i++) {
    aiFace face = assimpMesh->mFaces[i];
    assert(face.mNumIndices == 3);
    for (int j = 0; j < 3; j++) {
      mesh.triangles[i].vertices[j].point[0] = vertices[face.mIndices[j]].x;
      mesh.triangles[i].vertices[j].point[1] = vertices[face.mIndices[j]].y;
      mesh.triangles[i].vertices[j].point[2] = vertices[face.mIndices[j]].z;

      mesh.triangles[i].vertices[j].normal[0] = normals[face.mIndices[j]].x;
      mesh.triangles[i].vertices[j].normal[1] = normals[face.mIndices[j]].y;
      mesh.triangles[i].vertices[j].normal[2] = normals[face.mIndices[j]].z;
    }
  }

  return mesh;
}
