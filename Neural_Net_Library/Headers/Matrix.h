//
//  Matrix.h
//  Neural_Net_Library
//
//  Created by Thiviyan Saravanamuthu on 28.02.21.
//

#ifndef Matrix_h
#define Matrix_h

#include <iostream>
#include <vector>


namespace NN
{
    enum ActivationFunctions
    {
        RELU,
        D_RELU,
        SIGMOID,
        D_SIGMOID
    };


    class Matrix
    {
    public:
        
        //MARK: Constructor and Destructor
        Matrix(int rows, int cols);
        ~Matrix();
        
        
        //MARK: Acces to Elements / Mathematical stuff
        
        //Operators
        float &operator () (int m, int n) const {
            
            if(m >= rows || n >= cols)
            {
                std::cout << "Index Out of Bounds!!" << std::endl;
                std::cout << "Requested Index:" << rows << " and " << cols << std::endl;
                return Vals[0][0];
            }
            else
            {
                return Vals[m][n];
            }
            
        }
        
        void operator = (const Matrix M){
            if(this->rows == M.rows && this->cols == M.cols)
            {
                for(int n = 0; n < this->cols; n++)
                {
                    for(int j = 0; j < this->rows; j++)
                    {
                        Vals[n][j] = M.operator()(n, j);
                    }
                }
            }
        };
        
      
        Matrix operator * ( Matrix& SecondMatrix)
        {
            Matrix MultipliedMatrix(this->rows, SecondMatrix.cols);
            
            if(this->cols == SecondMatrix.rows && this->rows != SecondMatrix.cols)
            {
                
                for(int RowMatrixOne = 0; RowMatrixOne < this->rows; RowMatrixOne++)
                {
                    for(int SharedDimension = 0; SharedDimension < this->cols; SharedDimension++)
                    {
                      
                        for(int ColMatrixTwo = 0; ColMatrixTwo < SecondMatrix.getCols(); ColMatrixTwo++)
                        {
                                MultipliedMatrix(RowMatrixOne, ColMatrixTwo) += Vals[RowMatrixOne][SharedDimension] * SecondMatrix(SharedDimension, ColMatrixTwo);
                        }
                    }
                }
            }
            else if(this->cols == SecondMatrix.cols && this->rows == SecondMatrix.rows)
            {
                for(int col = 0; col < this->cols; col++)
                {
                    for(int row = 0; row < this->rows; row++)
                    {
                        MultipliedMatrix(col, row) = Vals[col][row] * SecondMatrix(col, row);
                    }
                }
            }
            
            return MultipliedMatrix;
        }
        
    
        
        
        //Scalar Multiplication
        void MultiplyByScalar(float Scalar);
        
        
        
        //Return Transposed Matrix; ex. 3 x 2 -> 2 x 3
        Matrix* GetTransposedMatrix();
        
        
        //Random Weight Initalization
        void RandomWeightInit();
        
        
        //Activations
        void ActivateNeurons(ActivationFunctions AF);
        float Sigmoid(float x);
        float D_Sigmoid(float x);
        float Relu(float x);
        float D_Relu(float x);
        
        
        //MARK: GET functions

        int getCols(){return this->cols;};
        int getRows(){return this->rows;};
        
        
    private:
        
        //MARK: Matrix Properties
        float** Vals;
        int rows;
        int cols;
        
    };
    



}


#endif /* Matrix_h */