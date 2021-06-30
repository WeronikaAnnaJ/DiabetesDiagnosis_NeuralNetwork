package DiabetesDiagnosis;

public class Neuron {

    public static double[] getOutputVector(double [][] weightMatrix, double [] inputVector, double [] bias){
        double outputVector[]= new double[weightMatrix.length];
       // System.out.println("weight matrix lenght= " + weightMatrix.length);
        for(int i =0 ; i < weightMatrix.length ;i ++){
            for(int j=0 ; j< weightMatrix[i].length;j++){
                outputVector[i]+=weightMatrix[i][j]*inputVector[j];
            }
        //    System.out.println( "outputvextoR [ "+i +"]" + outputVector[i]);
        }

        for(int i =0 ; i < weightMatrix.length ;i ++){
            outputVector[i]+=bias[i];
        }
        return outputVector;
    }

    public static double[] transformWithUnipolarSigmoidFunction(double[] vector, double lambda){
        //Z reguły lambda(0,1]
        double[] transfomedVector= new double[vector.length];
        for(int i=0 ; i < vector.length ; i++){
            transfomedVector[i]=1/( 1 + Math.exp( -lambda * vector[i] ) );
        }
        return transfomedVector;
    }



    public static double[] transformWithBipolarSigmoidFunction(double[] vector, double lambda){
        //Z reguły lambda(0,1]
        double[] transfomedVector= new double[vector.length];
        for(int i=0 ; i < vector.length ; i++){
            transfomedVector[i]=(2/( 1 + Math.exp( -lambda * vector[i] ) )-1);
        }
        return transfomedVector;
    }


    public static double[] transformWithUnipolarStepFunction(double[] vector){
        //Z reguły lambda(0,1]
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


    public static double [] determineWeightsForNeuron(double [] oldWeughts, double learningRate, double error, double[] inputVector ){
        double newWeights[]= new double[oldWeughts.length];
        for(int i =0 ; i < inputVector.length ; i++){
            newWeights[i]= learningRate * error * inputVector[i];
        }
        for(int i =0 ; i < oldWeughts.length ; i++){
            newWeights[i]+=oldWeughts[i];
        }
        return newWeights;
    }



    public static double determineNewBiasForNeuron(double oldBias, double learningRate, double error){
        return oldBias + (learningRate * error);
    }


    public static double determineErrorFor0utputNeuron(double expectedValue, double actualValue, double lambda, double outputValue){
        return (expectedValue-actualValue) * lambda * outputValue * (1-outputValue);
    }


    public static double determineErrorFor0utputNeuron(double expectedValue,  double actualValue){
        return (expectedValue-actualValue) ;
    }

    public static double determineErrorForHiddenNeuron(double[] errorNextLayer, double[] weight, double value, double lambda){
        double error=0.0;
        for(int i =0 ; i< errorNextLayer.length;i++){
            error+= errorNextLayer[i] * weight[i];
        }
        error*= value * lambda * (1 - value);
        return error;
    }

    public static double determineErrorForHiddenNeuronBipolar(double[] errorNextLayer, double[] weight, double value, double lambda){
        double error=0.0;
        for(int i =0 ; i< errorNextLayer.length;i++){
            error+= errorNextLayer[i] * weight[i];
        }
        error*=  value * (lambda/2) * (1 - Math.pow(value,2));//// is value
        return error;
    }



}
