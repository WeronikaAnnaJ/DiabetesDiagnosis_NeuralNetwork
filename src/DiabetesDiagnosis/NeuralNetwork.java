package DiabetesDiagnosis;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;



enum ActivationFunctions {
    UNIPOLAR_SIGMOID_FUNCTION,
    UNIPOLAR_STEP_FUNCTION,

}

public class NeuralNetwork {


    private List<String[]> learningDataSet= new ArrayList<>();
    //8 features
    private List<double[]> learningDataSetFeatures= new ArrayList<>();
    //decision
    private List<Double> learningDataSetDecisions=new ArrayList<>();


    private  List<String[]> testDataSet= new ArrayList<>();
    //8 features
    private List<double[]> testDataSetFeatures= new ArrayList<>();
    //decision
    private List<Double> testDataSetDecisions= new ArrayList<>();


    NeuralNetwork(List<String[]> learningDataSet, List<String[]>testDataSet){

        for (String [] set:learningDataSet) {
            int columnNumber=set.length;
            double [] features= new double[columnNumber-1];

            for(int i=0 ; i< columnNumber-1 ;i ++ ){
                features[i]= Double.parseDouble(set[i]);
            }
            learningDataSetFeatures.add(features);
        }

        for (String [] set:learningDataSet) {
            int columnNumber=set.length;
            double decision=0;
            decision= Double.parseDouble(set[columnNumber-1]);
            learningDataSetDecisions.add(decision);
        }

    }

    public void showLearningFeaturesDecisions(){
        int count=1;

        for (int i =0 ; i <learningDataSetFeatures.size() ; i ++ ) {
            System.out.print(count + ".  " );
            count++;
            double []array=learningDataSetFeatures.get(i);

            for(int j =0 ; j < array.length ; j++ ){
                System.out.print(array[j] + ", ");
            }
            System.out.println(" ----->  " + learningDataSetDecisions.get(i));
            System.out.println();
        }
    }

    public double[] getOutputVector(double [][] weightMatrix, double [] inputVector){
        //wylosować wagi początkowe dla wektorów
        //check if its possible to multiply
        double outputVector[]= new double[weightMatrix.length];
        System.out.println("weight matrix lenght= " + weightMatrix.length);

        for(int i =0 ; i < weightMatrix.length ;i ++){

            for(int j=0 ; j< weightMatrix[i].length;j++){
                outputVector[i]+=weightMatrix[i][j]*inputVector[j];
            }
            System.out.println( "outputvextoR [ "+i +"]" + outputVector[i]);
        }
        return outputVector;

    }

    public double[] transformWithUnipolarSigmoidFunction(double[] vector, double lambda){
        //Z reguły lambda(0,1]
        double[] transfomedVector= new double[vector.length];
        for(int i=0 ; i < vector.length ; i++){
            transfomedVector[i]=1/( 1 + Math.exp( -lambda * vector[i] ) );
        }
        return transfomedVector;
    }


    // use linear function
    public double[] transformWithUnipolarStepFunction(double[] vector){
        double[] transfomedVector= new double[vector.length];
                for(int i=0 ; i < vector.length ; i++){
                    if(vector[i]>= 0){
                        transfomedVector[i]=1;
                    }else{
                        transfomedVector[i]=0;
                    }
                }
        return transfomedVector;
    }

    //n >0learning rate, n=1
    //expected value
    //actual value
    //learning rate
    //old weight
    //new weight
    //output value
    //lambda
    //for unipolat sigmoid function
    public double determineErrorFor0utputNeuron(double expectedValue, double actualValue, double lambda, double outputValue){
        return (expectedValue-actualValue) * lambda * outputValue * (1-outputValue);
    }

    public double determineErrorForHiddenNeuron(double[] errorNextLayer, double[] weight, double value, double lambda){
        double error=0.0;
        for(int i =0 ; i< errorNextLayer.length;i++){
            error+= errorNextLayer[i] * weight[i];
        }
        error*= value * lambda * (1 - value);
        return error;
    }


    //old wetghts
    //input vector

    public double [] determineWeightsForNeuron(double [] oldWeughts, double learningRate, double error, double[] inputVector ){
        double newWeights[]= new double[oldWeughts.length];
        for(int i =0 ; i < inputVector.length ; i++){
            newWeights[i]= learningRate * error * inputVector[i];
        }
        for(int i =0 ; i < oldWeughts.length ; i++){
            newWeights[i]+=oldWeughts[i];
        }
        return newWeights;
    }



    public double determineNewBiasForNeuron(double oldBias, double learningRate, double error){
        return oldBias + (learningRate * error);
    }



    public static double[][] getRandomWeightsMatrix(int inputVectorLenght, int numberOfNeurons ){
        //range [-1,1]
        //rnge lenght 1 - (-1)= 2 max- min
        double weightsMatrix[][]= new double[numberOfNeurons][inputVectorLenght];
        for( int i =0 ; i < numberOfNeurons; i++){
            for( int j=0 ; j< inputVectorLenght ; j++){
                Random random = new Random();
                double randomValue = (random.nextDouble() * 2) - 1;
                weightsMatrix[i][j]=randomValue;
            }
        }
        return  weightsMatrix;
    }


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



}
