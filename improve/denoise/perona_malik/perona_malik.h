#pragma once
#include <iostream>
#include <cmath>
#include <omp.h>
#include <exception>
#include "..\..\Image.h"

using namespace std;
//Writen by Yngve Mardal Moe
//
//Functions used for Perona-Malik denoising as part of the PDImage python package
//This library also includes an accelerated iteration scheme that uses the acceleration
//scheme derived in [1] (ref. from [2]) as described in [2]. 
//
//[1] Y. E. Nesterov, "A method for solving the convex programming problem with convergence rate O(1/k^2)"
//	  Dokl. Akad. Nauk SSSR, 269 (1983), pp. 543–547(in Russian).
//[2] Beck, A and Teboulle, M. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." 
//	  SIAM journal on imaging sciences 2.1 (2009).


//Functions needed for Perona-Malik
double inline tukey(double x, double edge_level)
{
	//Computes weights for Perona-Malik iterations with Tukey's biweight norm
	edge_level = edge_level * sqrt(5.0);
	return (abs(x) <= edge_level) ? x*(1 - (x*x) / (edge_level*edge_level))*(1 - (x*x) / (edge_level*edge_level)) : 0;
}

double inline lorenz(double x, double edge_level)
{
	//Computes weights for Perona-Malik iterations with the Lorenzian error norm
	edge_level = edge_level / sqrt(2.0);
	return (2 * x) / (2 + ((x*x) / (edge_level*edge_level)) );
}

double inline isotropic(double x)
{
//Computes weights using isotropic diffusion/2-norm penality function
	return x;
}

double inline weight_function(double x, double edge_level, int method)
{
	switch(method)
	{
	case 1:
		return lorenz(x, edge_level);
		break;
	case 2:
		return tukey(x, edge_level);
		break;
	case 3:
		return isotropic(x);
	default:
		cout << "Invalid weight function, 1 for tukey, 2 for lorenz and 3 for isotropic diffusion";
		throw 255;

	}
}


//Functions used for automatic edge-level detection
void diff_2d_x(double* diff_x, double* image, int y, int x)
{
	//sets diff_x to be the x-derivative of image
	int i, j;
	for (i = 0; i < y; i++)
	{
		for (j = 0; j < x - 1; j++)
		{
			diff_x[i*(x-1) + j] = image[i*x + j + 1] - image[i*x + j];
		}
	}
}

void diff_2d_y(double* diff_y, double* image, int y, int x)
{
	//Sets diff_y to be the y-derivative of image
	int i, j;
	for (i = 0; i < y - 1; i++)
	{
		for (j = 0; j < x; j++)
		{
			diff_y[i*x + j] = image[(i + 1)*x + j] - image[i*x + j];
		}
	}
}

void norm_image_subtract_const(double* image, double* diff_x, double* diff_y, double c, int y, int x)
{
	//Computes sqrt((x-c)^2 + (y-c)^2) for all values in the images x=:diff_x: and y=:diff_y: and stores it in the variable :image:.

	for(int i = 0; i < y-1; ++i)
	{
		for(int j = 0; j < x-1; ++j)
		{
			image[i*(x-1) + j] = sqrt((diff_x[i*(x-1) + j] - c) * (diff_x[i*(x-1) + j] - c) + (diff_y[i*x + j] - c) * (diff_y[i*x + j] - c));
		}
	}
}

double auto_edge(double* image, int y, int x)
{
	double* diff_x = new double[y*(x-1)]();
	double* diff_y = new double[(y-1)*x]();
	double* diff_median = new double[(y-1)*(x-1)]();
	diff_2d_x(diff_x, image, y, x);
	diff_2d_y(diff_y, image, y, x);

	norm_image_subtract_const(diff_median, diff_x, diff_y, 0, y, x);

	double med = median(diff_median, y-1, x-1);
	norm_image_subtract_const(diff_median, diff_x, diff_y, med, y, x);
	med = 1.4826*median(diff_median, y-1, x-1);
	delete[] diff_x;
	delete[] diff_y;
	delete[] diff_median;
	
	return med;
}


//Gradient step for the Perona-Malik iterations
void gradient_step(double* current_iteration, double* previous_iteration, double edge_level, double step_length, int method, int y, int x)
{
	int i, j;
	//Top left pixel
	i = 0;
	j = 0;
	current_iteration[i*x + j] = previous_iteration[i*x + j]
							   + (step_length / 2)*(weight_function(previous_iteration[i*x + j + 1] - previous_iteration[i*x + j], edge_level, method)
													+ weight_function(previous_iteration[(i + 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	

	//Lower left pixel
	i = y - 1;
	j = 0;
	current_iteration[i*x + j] = previous_iteration[i*x + j]
							   + (step_length / 2)*(weight_function(previous_iteration[i*x + j + 1] - previous_iteration[i*x + j], edge_level, method)
													+ weight_function(previous_iteration[(i - 1)*x + j] - previous_iteration[i*x + j], edge_level, method));

	//Top right pixel
	i = 0;
	j = x - 1;
	current_iteration[i*x + j] = previous_iteration[i*x + j]
							   + (step_length / 2)*(weight_function(previous_iteration[i*x + j - 1] - previous_iteration[i*x + j], edge_level, method)
													+ weight_function(previous_iteration[(i + 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	
	//Bottom right pixel
	i = y - 1;
	j = x - 1;
	current_iteration[i*x + j] = previous_iteration[i*x + j]
							   + (step_length / 2)*(weight_function(previous_iteration[i*x + j - 1] - previous_iteration[i*x + j], edge_level, method)
													+ weight_function(previous_iteration[(i - 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
											 
	//Top row
	i = 0;
	for (j = 1; j < x-1; ++j)
	{
		current_iteration[i*x + j] = previous_iteration[i*x + j]
								   + (step_length / 3)*(weight_function(previous_iteration[i*x + j + 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[i*x + j - 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[(i + 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	}
	
	//Bottom row
	i = y - 1;
	for (j = 1; j < x-1; ++j)
	{
		current_iteration[i*x + j] = previous_iteration[i*x + j]
								   + (step_length / 3)*(weight_function(previous_iteration[i*x + j + 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[i*x + j - 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[(i - 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	}

	//Left row
	j = 0;
	for (i = 1; i < y-1; ++i)
	{
		current_iteration[i*x + j] = previous_iteration[i*x + j]
								   + (step_length / 3)*(weight_function(previous_iteration[i*x + j + 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[(i + 1)*x + j] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[(i - 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	}

	//Right row
	j = x-1;
	for (i = 1; i < y-1; ++i)
	{
		current_iteration[i*x + j] = previous_iteration[i*x + j]
								   + (step_length / 3)*(weight_function(previous_iteration[i*x + j - 1] - previous_iteration[i*x + j], edge_level, method)
														 + weight_function(previous_iteration[(i + 1)*x + j] - previous_iteration[i*x + j], edge_level, method)
														 + weight_function(previous_iteration[(i - 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	}
	
	//Non-border pixels

	for (i = 1; i < y-1; ++i)
	{
		for(j = 1; j < x-1; j++)
		current_iteration[i*x + j] = previous_iteration[i*x + j]
								   + (step_length / 4)*(weight_function(previous_iteration[i*x + j + 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[i*x + j - 1] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[(i + 1)*x + j] - previous_iteration[i*x + j], edge_level, method)
														+ weight_function(previous_iteration[(i - 1)*x + j] - previous_iteration[i*x + j], edge_level, method));
	}
}

void ext_perona_malik(double* image, double* raw, double edge_level, double step_length, int method, int max_it, int y, int x)
{
	if(edge_level <= 0)
	{
		edge_level = auto_edge(raw, y, x);
	}

	double* previous_iteration = new double[y*x]();
	copy_image(image, raw, y, x);

	for(int it = 0; it < max_it; ++it)
	{
		copy_image(previous_iteration, image, y, x);
		gradient_step(image, previous_iteration, edge_level, step_length, method, y, x);
	}
	delete[] previous_iteration;
}

void ext_fast_perona_malik(double* image, double* raw, double edge_level, double step_length, int method, int max_it, int y, int x)
{
	if (edge_level <= 0)
	{
		edge_level = 0.5*auto_edge(raw, y, x);
	}

	double* previous_iteration = new double[y*x]();
	double* momentum = new double[y*x]();
	double t = 1;
	double t1;

	copy_image(image, raw, y, x);
	copy_image(momentum, raw, y, x);
	for (int it = 0; it < max_it; ++it)
	{
		copy_image(previous_iteration, image, y, x);
		gradient_step(image, momentum, edge_level, step_length, method, y, x);
		t1 = t;
		t = (1 + sqrt(1 + 4 * t*t)) / 2;

		copy_image(momentum, image, y, x);											//momentum <- image
		subtract_image_second_from_first(momentum, previous_iteration, y, x);		//momentum <- momentum - previous_iteration
		multiply_image(momentum, ((t1 - 1) / t), y, x);								//momentum <- ((t1 - 1) / t)*momentum
		add_image(momentum, image, y, x);											//momentum <- image + momentum

	}
	delete[] momentum;
	delete[] previous_iteration;
	
}