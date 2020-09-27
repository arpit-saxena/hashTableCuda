#ifndef IMPORTER_H
#define IMPORTER_H

#include <string>

struct Triangle {
    float vertices[3][3];
};

struct Mesh {
    Triangle *triangles = nullptr;
    int numTriangles = 0;
};

Mesh import(std::string fileName);

#endif /* IMPORTER_H */