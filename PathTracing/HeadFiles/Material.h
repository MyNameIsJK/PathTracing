#pragma once
#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>
#include "GlobalDefine.h"

enum MaterialType
{
	DIFFUSE
};

class Material
{
public:
	MaterialType materialType;
	float3 emission;
	float ior;
	float3 kd, ks;
	float specularExponent;

public:
	Material(MaterialType t = DIFFUSE, float3 e = make_float3(0.0f));
	__device__ float3 eval(const float3& indir, const float3 outdir, const float3& normal)
	{
		switch (materialType)
		{
		case DIFFUSE:
		{
			float cosAlpha = dot(normal, outdir);
			if (cosAlpha > 0.0f)
			{
				float3 diffuse = kd / M_PI;
				return diffuse;
			}
			else
				return make_float3(0.0f);
			break;
		}
		default:
			return make_float3(0.0f);
			break;
		}
	}
};

#endif