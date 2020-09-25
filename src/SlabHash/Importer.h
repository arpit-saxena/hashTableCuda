#ifndef IMPORTER_H
#define IMPORTER_H

#include <string>

struct Point {
	int x;
	int y;
	int z;
};

struct Triangle {
	Point vertices[3];
};

struct Mesh {
	Triangle *triangles = nullptr;
	int numTriangles = 0;
};

Mesh import(std::string fileName);

#endif /* IMPORTER_H */