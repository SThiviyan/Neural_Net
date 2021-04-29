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
        D_SIGMOID,
        TANH,
        D_TANH
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
        }
        
        void operator = (const std::vector<float> M)
        {
            if(this->rows == M.size())
            {
                for(int n = 0; n < this->rows; n++)
                {
                  Vals[n][0] = M[n];
                }
            }
        }
      
        Matrix operator * ( Matrix& SecondMatrix)
        {
            Matrix MultipliedMatrix(this->rows, SecondMatrix.cols);
            
            if(this->cols == SecondMatrix.cols && this->rows == SecondMatrix.rows)
            {
                for(int row = 0; row < this->rows; row++)
                {
                    for(int col = 0; col < this->cols; col++)
                    {
                        MultipliedMatrix(row, col) = Vals[row][col] * SecondMatrix(row, col);
                    }
                }
            }
            else if(this->cols == 1 && SecondMatrix.rows == 1 && SecondMatrix.cols == 1)
            {
                for(int row = 0; row < this->rows; row++)
                {
                    for(int col = 0; col < 1; col++)
                    {
                            MultipliedMatrix(row, col) = Vals[row][col] * SecondMatrix(0, 0);
                    }
                }
                
            }
            else if(this->cols == SecondMatrix.rows)
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
           
            
            
            return MultipliedMatrix;
        }
        
        Matrix operator + ( Matrix& SecondMatrix)
        {
            Matrix NewMatrix = Matrix(this->rows, this->cols);

            if(this->rows == SecondMatrix.rows && this->cols == SecondMatrix.cols)
            {
                for (int n = 0; n < rows; n++) {
                    for (int j = 0; j < cols; j++) {
                        NewMatrix(n, j) = Vals[n][j] + SecondMatrix(n, j);
                    }
                }
               
            }
            else
            {
                std::cout << "Can't Add them together" << std::endl;
            }
            
            
            return NewMatrix;
        }
      
        Matrix operator - (Matrix& SecondMatrix)
        {
            Matrix NewMatrix = Matrix(this->rows, this->cols);
            
            if(this->rows == SecondMatrix.rows && this->cols == SecondMatrix.cols)
            {
                for (int n = 0; n < rows; n++) {
                    for (int j = 0; j < cols; j++) {
                        NewMatrix(n, j) = Vals[n][j] - SecondMatrix(n, j);
                    }
                }
               
            }
            else
            {
                std::cout << "Can't Subtract them" << std::endl;
            }
            
            
            return NewMatrix;
        }
        
        //Scalar Multiplication
        void MultiplyByScalar(float Scalar);
        void DivideByScalar(float Scalar);
        
        
        //Return Transposed Matrix; ex. 3 x 2 -> 2 x 3
        Matrix GetTransposedMatrix();
        
        
        //Random Weight Initalization
        void RandomWeightInit();
        
        
        //Activations
        void ActivateNeurons(ActivationFunctions AF);
        void TakeDerivative(ActivationFunctions AF);
        float Sigmoid(float x);
        float D_Sigmoid(float x);
        float Relu(float x);
        float D_Relu(float x);
        float tanh(float x);
        float dtanh(float x);
        
        
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
