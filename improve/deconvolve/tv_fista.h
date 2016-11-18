#pragma once
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>
//Written by Yngve Mardal Moe
//
//Functions that are used in TV denoising and TV deconvolution as part of a python package
//
// [1]: Beck, A & Teboulle, M - Fast Gradient-Based Algorithms for Constrained
//      Total Variation Image Denoising and Deblurring Problems.
//      2009, IEEE Transactions on Image Processing.
// [2]: O'Donoghue, B & Candes, E - Adaptive Restart for Accelerated Gradient Schemes
//   	2012, Foundations of Computational Mathematics

using namespace std;


void multiply_image(double* image, double factor, int y, int x)
{
	//Multiplies all elements of :image: by factor
	int i;
#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image[i] *= factor;
	}
}

void add_image(double* image1, double* image2, int y, int x)
{
	//Adds the images :image1: and :image2: together and stores it in :image1:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image1[i] += image2[i];
	}
}

void subtract_image_second_from_first(double* image1, double* image2, int y, int x)
{
	//Subtracts the elements of :image2: from :image1: and stores it in :image1:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image1[i] -= image2[i];
	}
}

void subtract_image_first_from_second(double* image1, double* image2, int y, int x)
{
	//Subtracts the elements of :image1: from :image2: and stores it in :image1:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image1[i] = image2[i] - image1[i];
	}
}

void set_zero(double* image, int y, int x)
{
	//Set all the elements of :image: to zero
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image[i] = 0;
	}
}

void copy_image(double* image, double* raw, int y, int x)
{
	//Copy all elements of :raw: into :image:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image[i] = raw[i];
	}
}



//Projection functions used in the FISTA iterations and proximal gradient step computations
void dual_projection(double* dual_x, double* dual_y, int y, int x)
{
	//Projects the dual of the ROF functional onto its feasible set
	int i;
	double dual_norm;


#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
			dual_norm = max(1., sqrt(dual_x[i] * dual_x[i] +
								dual_y[i] * dual_y[i]));
			dual_x[i] /= dual_norm;
			dual_y[i] /= dual_norm;
	}

}

void primal_projection(double* image, double min_value, double max_value, int y, int x)
{
	//Projects the ROF functional onto its feasible set
	int i;


#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		if (image[i] < min_value)
		{
			image[i] = min_value;
		}
		else if (image[i] > max_value)
		{
			image[i] = max_value;
		}
	}
}

void primal_off_projection(double* image, double* raw, double gamma, double min_value, double max_value, int y, int x)
{
	//The projection off the feasible set for the ROF functional
	int i;

//#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		if (image[i] < min_value)
		{
			image[i] -= -min_value;
		}
		else if (image[i] > max_value)
		{
			image[i] -= -max_value;
		}
		else
		{
			image[i] = 0;
		}
	}
}



//Functions used to compute the proximal gradient steps of the minimization target functions
//for the ROF problem. They are also used to go from the dual to primal problem.
void sub_diff_2d_x(double* dual_x, double* image, int y, int x)
{
	//Subtracts the forward finite difference x-derivative of image to dual_x
	int i, j;
	for (i = 0; i < y; i++)
	{
		for (j = 0; j < x - 1; j++)
		{
			dual_x[i*x + j] += image[i*x + j] - image[i*x + j + 1];
		}
	}
}

void sub_diff_2d_y(double* dual_y, double* image, int y, int x)
{
	//Subtracts the forward finite difference x-derivative of image to dual1
	int i, j;
	for (i = 0; i < y - 1; i++)
	{
		for (j = 0; j < x; j++)
		{
			dual_y[i*x + j] += image[i*x + j] - image[(i + 1)*x + j];
		}
	}
}

void gradient_2d(double* dual_x, double* dual_y, double* image, int y, int x)
{
	//Adds the forward finite-difference derivative of image in the x-direction to dual_x and in the y-direction to dual_y
	sub_diff_2d_x(dual_x, image, y, x);
	sub_diff_2d_y(dual_y, image, y, x);
}

void divergence_2d(double* image, double* dual_x, double* dual_y, int y, int x)
{
	//Adds the backwards finite difference divergence of the dual function,
	//where dual_x is the x-component of dual and dual_y is the y-component of dual, to image
	int i, j, index;
	for (i = 0; i < y; i++)
	{
		for (j = 0; j < x; j++)
		{
			index = i*x + j;
			if (i != 0 && j != 0)
			{
				image[index] += dual_x[index] + dual_y[index] - dual_x[index - 1] - dual_y[index - x];
			}
			else if (i == 0 && j != 0)
			{
				image[index] += dual_x[index] + dual_y[index] - dual_x[index - 1];
			}
			else if (i != 0 && j == 0)
			{
				image[index] += dual_x[index] + dual_y[index] - dual_y[index - x];
			}
			else
			{
				image[index] += dual_x[index] + dual_y[index];
			}
		}
	}
}



//Functions used in the computation of the minimization target functions for the ROF problem
double sum_squares(double* image, int y, int x)
{
	//Computes the square of the 2-norm of :image:
	int i;
	double norm = 0;

#pragma omp parallel for
	for (i = 0; i < x*y; i++)
	{
		norm += image[i] * image[i];
	}
	return norm;
}

double TV_norm(double* image, int y, int x)
{
	//Computes the TV_norm of :image:
	int i;
	double norm = 0;

	double* diff_x;
	double* diff_y;

	diff_x = (double*)calloc(x*y, sizeof(double));
	sub_diff_2d_x(diff_x, image, y, x);


	diff_y = (double*)calloc(x*y, sizeof(double));
	sub_diff_2d_y(diff_y, image, y, x);

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		norm += sqrt((diff_x[i] * diff_x[i])
					+(diff_y[i] * diff_y[i]));
	}

	free(diff_x);
	free(diff_y);

	return norm;
}

double sum_squared_error(double* image, double* raw, int y, int x)
{
	//Computes sum{(image[i]-raw[i])^2} for all i
	int i;
	double norm = 0;

#pragma omp parallel for
	for (i = 0; i < y; i++)
	{
		norm += (image[i] - raw[i]) * (image[i] - raw[i]);
	}
	return norm;
}




//The minimization target functions for the ROF problem
double TV_problem(double* image, double* raw, double gamma, int y, int x)
{
	//Returns the value of the TV deconvolution problem, min ||I - b||^2 + g*||I||_TV,
	//where b is the image we want to denoise, g is the regularization constant (noise level)
	//and I is the current iteration.
	//
	//:image: = I
	//:raw: = b
	//:gamma: = g
	//:y: = Number of vertical pixels
	//:x: = Number of horizontal pixels
	
	return sum_squared_error(image, raw, y, x) + gamma*TV_norm(image, y, x);
}

double TV_deconvolve_problem(double* image, double*decon_image, double*raw, double gamma, int y, int x)
{
	//Returns the value of the TV deconvolution problem, min ||F*I - b||^2 + g*||I||_TV, where F* is the convolution operator
	//we want to invert, b is the image we want to deconvolve, g is the regularization constant (noise level)
	//and I is the current iteration.
	//
	// :image: = I
	//:decon_image: = F*I
	//:raw: = b
	//:gamma: = g
	//:y: = Number of vertical pixels
	//:x: = Number of horizontal pixels
	
	return sum_squared_error(image, raw, y, x) + gamma*TV_norm(decon_image, y, x);
}

double dual_problem(double* dual_x, double* dual_y, double* image, double* raw, double gamma,
	double min_value, double max_value, int y, int x)
{
	//Returns the value of the dual problem described in [1].
	double problem_value = 0;

	set_zero(image, y, x);
	divergence_2d(image, dual_x, dual_y, y, x);
	multiply_image(image, gamma, y, x);
	subtract_image_first_from_second(image, raw, y, x);

	problem_value += sum_squares(image, y, x);
	primal_off_projection(image, raw, gamma, min_value, max_value, y, x);
	problem_value += sum_squares(image, y, x);
	return problem_value;
}




//The function to get from the dual variables to primal variables
void dual_to_image(double* image, double* dual_x, double* dual_y, double* raw, double gamma,
	double min_value, double max_value, int y, int x)
{
	//Computes what image to create from the dual variables, and stores it in :*image:, using equation 4.4 from [1]
	set_zero(image, y, x);
	divergence_2d(image, dual_x, dual_y, y, x);
	multiply_image(image, gamma, y, x);
	subtract_image_first_from_second(image, raw, y, x);
	primal_projection(image, min_value, max_value, y, x);
}

//The proximal gradient step for the ROF dual
void proximal_step(double* dual_x, double* dual_y, double* image, double* raw, double gamma,
	double min_value, double max_value, int y, int x)
{
	//Performs a proximal step of the dual variable as described in [1].

	dual_to_image(image, dual_x, dual_y, raw, gamma, min_value, max_value, y, x);
	multiply_image(image, 1 / (8 * gamma), y, x);
	gradient_2d(dual_x, dual_y, image, y, x);
	dual_projection(dual_x, dual_y, y, x);
}


//The FISTA function.
void TV_FISTA(double* image, double* raw, double gamma, double min_value, double max_value,
	int max_it, double eps, int y, int x)
{
	//Performs FISTA iterations for the TV problem, with a scheme for restarting the momentum term as described in [1] and [2]
	double* dual_xn;
	double* dual_yn;
	double* dual_xn_1;
	double* dual_yn_1;
	double* dual_x_momentum;
	double* dual_y_momentum;
	double* temp_x;
	double* temp_y;
	double t = 1;
	double t_1 = 1;
	double fn;
	double fn_1;
	int n;

	dual_xn = (double*)calloc(x*y, sizeof(double));
	dual_yn = (double*)calloc(x*y, sizeof(double));
	dual_xn_1 = (double*)calloc(x*y, sizeof(double));
	dual_yn_1 = (double*)calloc(x*y, sizeof(double));
	dual_x_momentum = (double*)calloc(x*y, sizeof(double));
	dual_y_momentum = (double*)calloc(x*y, sizeof(double));

	fn = dual_problem(dual_xn, dual_yn, image, raw, gamma, min_value, max_value, y, x);
	for (n = 0; n < max_it; n++)
	{
		fn_1 = fn;
		// Set previous variable
		copy_image(dual_xn_1, dual_xn, y, x);		//xn-1 = xn
		copy_image(dual_yn_1, dual_yn, y, x);

		// Update current variable
		proximal_step(dual_x_momentum, dual_y_momentum, image, raw, gamma, min_value, max_value, y, x);
		temp_x = dual_xn;							//temp_x points to previous iteration
		temp_y = dual_yn;
		dual_xn = dual_x_momentum;					//dual_xn points to current iteration
		dual_yn = dual_y_momentum;
		dual_x_momentum = temp_x;					//dual_x_momentum points to temp_x, which points to previous iteration
		dual_y_momentum = temp_y;
		fn = dual_problem(dual_xn, dual_yn, image, raw, gamma, min_value, max_value, y, x);

		// Test for reseting the momentum
		if(fn_1 - fn < 0)
		{
			t = 1;
			t_1 = 1;
		}
		// Test for convergence
		else if (fn_1 - fn < eps*fn)
		{
			break;
		}


		// Update momentum multiplier
		t_1 = t;
		t = (1 + sqrt(1 + 4*t*t)) / 2;

		// Update momentum term
		copy_image(dual_x_momentum, dual_xn, y, x);		// dual_x_m = dual_xn_1
		copy_image(dual_y_momentum, dual_yn, y, x);
		subtract_image_second_from_first(dual_x_momentum, dual_xn_1, y, x);	//dual_x_m = dual_xn - dual_xn_1
		subtract_image_second_from_first(dual_y_momentum, dual_yn_1, y, x);
		multiply_image(dual_x_momentum, (t_1 - 1) / t, y, x);	//r = (t_1 - 1 / t) * (pn - pn_1)
		multiply_image(dual_y_momentum, (t_1 - 1) / t, y, x);

		add_image(dual_x_momentum, dual_xn, y, x);	//r_n = pn + (t_1 - 1 / t) * (pn - pn_1)
		add_image(dual_y_momentum, dual_yn, y, x);


	}
	temp_x = NULL;
	temp_y = NULL;

	dual_to_image(image, dual_xn, dual_yn, raw, gamma, min_value, max_value, y, x);
	free(dual_xn);
	free(dual_yn);
	free(dual_xn_1);
	free(dual_yn_1);
	free(dual_x_momentum);
	free(dual_y_momentum);
}
