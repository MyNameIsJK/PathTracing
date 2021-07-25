#pragma once
#ifndef LIGHT_H
#define LIGHT_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_math.h>

class Light
{
public:
	float3 position;
	float3 intensity;

	Light(const float3& p, const float3& i);
	virtual ~Light() = default;
};

#endif