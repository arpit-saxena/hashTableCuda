#ifndef IMPORTER_H
#define IMPORTER_H

#include <string>
#include <glm/gtc/type_ptr.hpp>

struct Vertex {
    float point[3];
    float normal[3];
    float hasCollided;     // 0.0 if not collided, 1.0 if collided
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