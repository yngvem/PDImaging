#include <iostream>
#include <cmath>
#include <omp.h>
//Functions that are integral to the image processing functions
//in the PDImaging python toolbox written by Yngve Mardal Moe.

//Basic image mathematics
void inline multiply_image(double* image, double factor, int y, int x)
{
	//Multiplies all elements of :image: by factor
	int i;
#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image[i] *= factor;
	}
}

void inline add_image(double* image1, double* image2, int y, int x)
{
	//Adds the images :image1: and :image2: together and stores it in :image1:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image1[i] += image2[i];
	}
}

void inline subtract_image_second_from_first(double* image1, double* image2, int y, int x)
{
	//Subtracts the elements of :image2: from :image1: and stores it in :image1:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image1[i] -= image2[i];
	}
}

void inline subtract_image_first_from_second(double* image1, double* image2, int y, int x)
{
	//Subtracts the elements of :image1: from :image2: and stores it in :image1:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image1[i] = image2[i] - image1[i];
	}
}

void inline set_zero(double* image, int y, int x)
{
	//Set all the elements of :image: to zero
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image[i] = 0;
	}
}

void inline copy_image(double* image, double* raw, int y, int x)
{
	//Copy all elements of :raw: into :image:
	int i;

#pragma omp parallel for
	for (i = 0; i < y*x; i++)
	{
		image[i] = raw[i];
	}
}


//Merge sort algorithm
void merge(double* first_half, int first_length, double* second_half, int second_length, double* full_array)
{
	int l = 0;
	int r = 0;
	for (int i = 0; i < first_length + second_length; ++i)
	{
		if (l >= first_length)
		{
			full_array[i] = second_half[r];
			r++;
		}
		else if (r >= second_length)
		{
			full_array[i] = first_half[l];
			l++;
		}
		else if (first_half[l] <= second_half[r])
		{
			full_array[i] = first_half[l];
			l++;
		}
		else
		{
			full_array[i] = second_half[r];
			r++;
		}
	}
}
void merge_sort(double* sort_array, int length)
{
	if (length < 2)
	{
		return;
	}
	else
	{
		double* temp = new double[length]();
		int center = length / 2;
		merge_sort(sort_array, center);
		merge_sort(sort_array + center, length - center);
		merge(sort_array, center, sort_array + center, length - center, temp);
		for (int i = 0; i < length; i++)
		{
			sort_array[i] = temp[i];
		}
	}
}


//Image statistics
double median(double* image, int y, int x)
{
	double* temp_image = new double[y*x]();
	copy_image(temp_image, image, y, x);
	merge_sort(temp_image, y*x);
	double med = ((y*x) % 2 == 1) ? temp_image[(y*x) / 2] : (temp_image[(y*x) / 2] + temp_image[(y*x) / 2 - 1]) / 2.0;
	delete[] temp_image;
	return med;
}

double sum_squares(double* image, int y, int x)
{
	//Computes the square of the 2-norm of :image:
	int i;
	double norm = 0;

	for (i = 0; i < x*y; i++)
	{
		norm += image[i] * image[i];
	}
	return norm;
}

double sum_squared_error(double* image, double* raw, int y, int x)
{
	//Computes sum{(image[i]-raw[i])^2} for all i
	int i;
	double norm = 0;

	for (i = 0; i < y; i++)
	{
		norm += (image[i] - raw[i]) * (image[i] - raw[i]);
	}
	return norm;
}