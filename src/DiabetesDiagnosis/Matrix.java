package DiabetesDiagnosis;

import java.util.Random;

public abstract class Matrix {


    public static void showMatrix( double[][] matrix){
        System.out.println();
        for( int i =0 ; i < matrix.length ; i++){
            for (int j =0 ; j< matrix[i].length ;j ++){
                System.out.print(matrix[i][j] +"  ");
            }
            System.out.println();
        }
    }


    public static void showMatrix( double[]matrix){
        System.out.println();
        for( int i =0 ; i < matrix.length ; i++){
            System.out.print(matrix[i]+ "  ");
        }
    }



    public static double[][] getRandomWeightsMatrix(int inputVectorLenght, int numberOfNeurons ){
        //range [-0.1,0.1]
        double weightsMatrix[][]= new double[numberOfNeurons][inputVectorLenght];
        for( int i =0 ; i < numberOfNeurons; i++){
            for( int j=0 ; j< inputVectorLenght ; j++){
                Random random = new Random();
                double randomValue = (random.nextDouble() * 0.2) - 0.1;
                weightsMatrix[i][j]=randomValue;
            }
        }
        return  weightsMatrix;
    }






}
