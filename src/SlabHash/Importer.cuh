#ifndef IMPORTER_H
#define IMPORTER_H

#include <string>
#include <glm/gtc/type_ptr.hpp>

struct Vertex {
    float point[3];
    float normal[3];
};

struct Triangle {
    Vertex vertices[3];
};

struct Mesh {
    Triangle *triangles = nullptr;
    int numTriangles = 0;
};

Mesh import(std::string fileName);

#endif /* IMPORTER_H */