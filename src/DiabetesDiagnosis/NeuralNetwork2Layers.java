package DiabetesDiagnosis;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.pow;

public class NeuralNetwork2Layers {



    private List<double[]> learningDataSetFeatures;
    private List<Double> learningDataSetDecisions;

    private List<double[]> learningDatataSetResults = new ArrayList<>();

    int correctResult=0;

    private double[] inputVector;
    private double lambda;
    private double learningRate;


    private double[][] weightsHiddenLayer;
    private double[][] weightsHiddenLayer2;
    private double[][] weightsOutputLayer;


    private double[] biasHiddenLayer;
    private double[] biasHiddenLayer2;

    private double[] biasOutputLayer;

    private double[] net1;
    private double[] Y1;
    private double[] net1_2;
    private double[] Y1_2;

    private double[] net2;
    private double[] Y2;

    private double[] hiddenLayerError;
    private double[] hiddenLayerError2;
    private double[] hiddenLayerError3;
    private double[] outpuLayerError;


    public List<double[]> meanSquaredErrorBefore = new ArrayList<>();
    public List<double[]> meanSquaredErrorAfter = new ArrayList<>();


    NeuralNetwork2Layers(List<double[]> learningDataSetFeatures, List<Double> learningDataSetDecisions) {
        this.learningDataSetFeatures = learningDataSetFeatures;
        this.learningDataSetDecisions = learningDataSetDecisions;
    }


    public void calculateOutputForNetwork(double[] inputVector) {
        net1 = Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
        Y1 = Neuron.transformWithBipolarSigmoidFunction(net1, lambda);

        net1_2 = Neuron.getOutputVector(weightsHiddenLayer2, Y1, biasHiddenLayer2);
        Y1_2 = Neuron.transformWithBipolarSigmoidFunction(net1_2, lambda);


        net2 = Neuron.getOutputVector(weightsOutputLayer, Y1_2, biasOutputLayer);
        //change function
        Y2 = Neuron.transformWithUnipolarStepFunction(net2);
    //    System.out.println("y2-> "+ Y2[0]);
        //for testing
        learningDatataSetResults.add(Y2);
    }



    public void calculateErrorsForLayers(double []expectedValuesOutputLayer ){
        outpuLayerError= new double[weightsOutputLayer.length];
        for(int i =0 ; i < outpuLayerError.length; i++){
            outpuLayerError[i]=Neuron.determineErrorFor0utputNeuron(expectedValuesOutputLayer[i],Y2[i] );
           // System.out.println(outpuLayerError[i]);
        }


        hiddenLayerError2= new double[weightsHiddenLayer2.length];
        for (int i =0 ; i < hiddenLayerError2.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsOutputLayer.length];
            for(int j=0 ; j< weightsForNextNeuron.length; j++){
                weightsForNextNeuron[j]=weightsOutputLayer[j][i]; // ?? ERROR
            }
            hiddenLayerError2[i]=Neuron.determineErrorForHiddenNeuronBipolar(outpuLayerError,weightsForNextNeuron,Y1_2[i], lambda);
        }

        hiddenLayerError= new double[weightsHiddenLayer.length];
        for (int i =0 ; i < hiddenLayerError.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsHiddenLayer2.length];
            for(int j=0 ; j< weightsForNextNeuron.length; j++){
                weightsForNextNeuron[j]= weightsHiddenLayer2[j][i]; // ?? ERROR
            }
            hiddenLayerError[i]=Neuron.determineErrorForHiddenNeuronBipolar(hiddenLayerError2,weightsForNextNeuron,Y1[i], lambda);
        }

    }




    public double [] calculateNewBiasForHiddenLayer(double[] biasHiddenLayer,double [] hiddenLayerError){
        double[] newbiasHiddenLayer= new double[biasHiddenLayer.length];
        for (int i=0 ; i < newbiasHiddenLayer.length ; i ++){
            newbiasHiddenLayer[i]=Neuron.determineNewBiasForNeuron(biasHiddenLayer[i],learningRate,hiddenLayerError[i]);

        }
     //   Matrix.showMatrix(newbiasHiddenLayer);
        return newbiasHiddenLayer;
    }


    public void calculateNewBiasForOutputLayer(){
        double[] newBiasOutputLayer= new double[biasOutputLayer.length];
        for (int i=0 ; i < newBiasOutputLayer.length ; i ++){
            newBiasOutputLayer[i]=Neuron.determineNewBiasForNeuron(biasOutputLayer[i],learningRate,outpuLayerError[i]);
        }
        this.biasOutputLayer=newBiasOutputLayer;
    }






    public void calculateWeightsForOutputLayer(){
        double [][] newWeightsForOutputNeurons= new double[weightsOutputLayer.length][];
        for(int i=0 ; i< newWeightsForOutputNeurons.length ; i ++){
            double [] oldWeightsForOutputNeuron= new double[weightsOutputLayer[i].length];
            for(int j=0; j<weightsOutputLayer[i].length; j++){
                oldWeightsForOutputNeuron[j]=weightsOutputLayer[i][j];
            }
         //   System.out.println("old weights for neuron : ");
        //    Matrix.showMatrix(oldWeightsForOutputNeuron);

            double[] newWeightsForOutputNeuron= Neuron.determineWeightsForNeuron(oldWeightsForOutputNeuron,learningRate,outpuLayerError[i],Y1_2);
      //      System.out.println("new weights for neuron : ");
      //      Matrix.showMatrix(newWeightsForOutputNeuron);

            newWeightsForOutputNeurons[i]=newWeightsForOutputNeuron;
        }
        this.weightsOutputLayer=newWeightsForOutputNeurons;
     //   Matrix.showMatrix(newWeightsForOutputNeurons);
    }



    public double [][] calculateWeightsForHiddenLayer( double[][] weightsHiddenLayer, double[] hiddenLayerError, double [] inputVector){
        double [][] newWeightsForHiddenNeurons= new double[weightsHiddenLayer.length][];

        for(int i=0 ; i< newWeightsForHiddenNeurons.length ; i ++){
            double [] oldWeightsForHiddenNeuron= new double[weightsHiddenLayer[i].length];

            for(int j=0; j<weightsHiddenLayer[i].length; j++){
                oldWeightsForHiddenNeuron[j]=weightsHiddenLayer[i][j];
            }
         //   System.out.println("old weights for neuron : ");
         //   Matrix.showMatrix(oldWeightsForHiddenNeuron);

            double[] newWeightsForHiddenNeuron= Neuron.determineWeightsForNeuron(oldWeightsForHiddenNeuron,learningRate,hiddenLayerError[i],inputVector);
       //     System.out.println("new weights for neuron : ");
        //    Matrix.showMatrix(newWeightsForHiddenNeuron);

            newWeightsForHiddenNeurons[i]=newWeightsForHiddenNeuron;
        }
        return  newWeightsForHiddenNeurons;
    }








    public void carryOutEpoch( double [] input,double [] expectedValuesOutputLayer){
        calculateOutputForNetwork(input);
        calculateErrorsForLayers(expectedValuesOutputLayer);


        double [][] newWeightsForHiddenNeurons2= calculateWeightsForHiddenLayer(weightsHiddenLayer2, hiddenLayerError2, Y1);
        weightsHiddenLayer2=newWeightsForHiddenNeurons2;


        double [][] newWeightsForHiddenNeurons= calculateWeightsForHiddenLayer(weightsHiddenLayer, hiddenLayerError, input);
        weightsHiddenLayer=newWeightsForHiddenNeurons;


        calculateWeightsForOutputLayer();


        double [] newBiasHiddenLayer=calculateNewBiasForHiddenLayer(biasHiddenLayer,hiddenLayerError);
        biasHiddenLayer=newBiasHiddenLayer;

        double [] newBiasHiddenLayer2=calculateNewBiasForHiddenLayer(biasHiddenLayer2,hiddenLayerError2);
        biasHiddenLayer2=newBiasHiddenLayer2;


        calculateNewBiasForOutputLayer();


    }






    public double calculateMeanSquaredError( ){
        double error=0;
        for(int i=0;i<learningDatataSetResults.size();i++){
         //  System.out.println(">>>>>>>>>>>>>>   expectedValues "+ learningDataSetDecisions.get(i) +"   >>>>>>>>>>  actualValue" +  learningDatataSetResults.get(i)[0]);
            error+=pow((learningDataSetDecisions.get(i)-learningDatataSetResults.get(i)[0]),2);
            if(learningDataSetDecisions.get(i)==learningDatataSetResults.get(i)[0]){
                correctResult++;
            }
    //       System.out.println("error " + i + " " + error );
        }
        error= error/2;
     //   System.out.println(">>>>>>>>>>> ERROR " + error);
        return error;

    }
















    public void setBiasOutputLayer(double[] biasOutputLayer) {
        this.biasOutputLayer = biasOutputLayer;
    }


    public void setBiasHiddenLayer2(double[] biasHiddenLayer2) {
        this.biasHiddenLayer2 = biasHiddenLayer2;
    }

    public void setBiasHiddenLayer(double[] biasHiddenLayer) {
        this.biasHiddenLayer = biasHiddenLayer;
    }


    public void setWeightsOutputLayer(double[][] weightsOutputLayer) {
        this.weightsOutputLayer = weightsOutputLayer;
    }




    public void setWeightsHiddenLayer2(double[][] weightsHiddenLayer2) {
        this.weightsHiddenLayer2 = weightsHiddenLayer2;
    }

    public void setWeightsHiddenLayer(double[][] weightsHiddenLayer) {
        this.weightsHiddenLayer = weightsHiddenLayer;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double[][] getWeightsHiddenLayer2() {
        return weightsHiddenLayer2;
    }



    public double[] getBiasOutputLayer() {
        return biasOutputLayer;
    }

    public double[] getBiasHiddenLayer() {
        return biasHiddenLayer;
    }

    public double[] getBiasHiddenLayer2() {
        return biasHiddenLayer2;
    }






    public double[] getNet1() {
        return net1;
    }


    public double[] getNet1_2() {
        return net1_2;
    }


    public double[] getNet2() {
        return net2;
    }



    public double[] getY1() {
        return Y1;
    }

    public double[] getY2() {
        return Y2;
    }

    public double[] getY1_2() {
        return Y1_2;
    }





    public double[] getHiddenLayerError() {
        return hiddenLayerError;
    }

    public double[] getHiddenLayerError2() {
        return hiddenLayerError2;
    }

    public double[] getHiddenLayerError3() {
        return hiddenLayerError3;
    }

    public double[] getOutpuLayerError() {
        return outpuLayerError;
    }




    public List<Double> getLearningDataSetDecisions() {
        return learningDataSetDecisions;
    }

    public List<double[]> getLearningDataSetFeatures() {
        return learningDataSetFeatures;
    }

    public void resetLearningDataSetResults() {
        learningDatataSetResults.removeAll(learningDatataSetResults);
    }

    public List<double[]> getLearningDatataSetResults() {
        return learningDatataSetResults;
    }

    public void resetCorrectResults(){
        correctResult=0;
    }

    public int getCorrectResult() {
        return correctResult;
    }



}
