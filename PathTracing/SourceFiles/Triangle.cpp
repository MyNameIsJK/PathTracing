#include "Triangle.h"
#include <cassert>
#include "GlobalFunc.h"
#include "OBJ_Loader.hpp"

Triangle::Triangle(float3 _v0, float3 _v1, float3 _v2, Material* m)
{
	v0 = _v0;
	v1 = _v1;
	v2 = _v2;
	e1 = v1 - v0;
	e2 = v2 - v0;
	normal = normalize(cross(e1, e2));
	area = 0.5f * length(cross(e1, e2));
	if (m != nullptr)
		material = *m;
}



float Triangle::getArea()
{
	return area;
}

MeshTriangle::MeshTriangle(const std::string& fileName, Material* m)
{
	objl::Loader loader;
	loader.LoadFile(fileName);
	assert(loader.LoadedMeshes.size() == 1);
	auto mesh = loader.LoadedMeshes[0];

	float3 maxVert = make_float3(FLT_MIN);
	float3 minVert = make_float3(FLT_MAX);

	float3 tmpv[3];
	for (int i = 0; i < mesh.Vertices.size(); i+=3)
	{
		for (int j = 0; j < 3; j++)
		{
			tmpv[j] = make_float3(mesh.Vertices[i + j].Position.X,
				mesh.Vertices[i + j].Position.Y,
				mesh.Vertices[i + j].Position.Z);
			minVert = fminf(minVert, tmpv[j]);
			maxVert = fmaxf(maxVert, tmpv[j]);
		}
		triangles.emplace_back(tmpv[0], tmpv[1], tmpv[2], m);
	}
}
