#ifndef SCENE_H_
#define SCENE_H_

#include <map>
#include <string>
#include <vector>
#include <assimp/Importer.hpp>

/*
 * Collection of meshes, to be displayed on the screen.
 */
class Scene {
	Assimp::Importer importer;
	map<String, Mesh> meshes;
	void importMesh(String file);

public:
	Scene(Vector<String> files); // Initialise scene with meshes specified by the files
};



#endif /* SCENE_H_ */
