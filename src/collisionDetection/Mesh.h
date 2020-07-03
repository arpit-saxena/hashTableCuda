#ifndef MESH_H_
#define MESH_H_

#include <vector_types.h>

class Mesh {
	float4 *vertices;

public:
	Mesh(std::string fileName);
	~Mesh();
};

#endif /* MESH_H_ */
