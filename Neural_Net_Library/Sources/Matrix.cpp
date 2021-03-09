//
//  Matrix.cpp
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 28.02.21.
//

#include "Matrix.h"
#include <stdio.h>
#include <time.h>


//MARK: Constructor and Destructor

NN::Matrix::Matrix(int rows, int cols)
{
    this->cols = cols;
    this->rows = rows;
    
    
    //Allocating Memory for Columns
    Vals = new float*[rows];
    
    for(int n = 0; n < rows; n++)
    {
        //Now for Rows
        Vals[n] = new float[cols];
    }
    
    
}

NN::Matrix::~Matrix()
{
   
}


//MARK: Mathematical Stuff

void NN::Matrix::MultiplyByScalar(float Scalar)
{
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; cols++) {
            Vals[row][col] = Vals[row][col] * Scalar;
        }
    }
    
}


void NN::Matrix::RandomWeightInit()
{
    srand(time(NULL));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Vals[row][col] = float(rand() % 100 + 1) / 100;
        }
    }
}


//MARK: ActivationFunction Stuff (Seperate from Math because...)

void NN::Matrix::ActivateNeurons(ActivationFunctions AF)
{
    switch (AF) {
        
        case SIGMOID:
           
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = Sigmoid(Vals[row][col]);
                }
            }
            
            break;
            
        case D_SIGMOID:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = D_Sigmoid(Vals[row][col]);
                }
            }
            
            break;
            
        case RELU:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = Relu(Vals[row][col]);
                }
            }
            
            break;
            
        case D_RELU:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = D_Relu(Vals[row][col]);
                }
            }
            
            break;
        default:
            break;
    }
}

//2 Activation functions and their derivatives

float NN::Matrix::Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float NN::Matrix::D_Sigmoid(float x)
{
    return Sigmoid(x) * (1 - Sigmoid(x));
}

float NN::Matrix::Relu(float x)
{
    if(x < 0)
    {
        return 0.f;
    }
    else
    {
        return x;
    }
}

float NN::Matrix::D_Relu(float x)
{
    if(x > 0.0f)
    {
        return 1;
    }
    return 0;
}
