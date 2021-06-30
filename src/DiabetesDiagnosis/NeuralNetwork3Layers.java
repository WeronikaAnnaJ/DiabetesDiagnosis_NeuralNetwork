package DiabetesDiagnosis;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.pow;

public class NeuralNetwork3Layers {


    private List<double[]> learningDataSetFeatures;
    private List<Double> learningDataSetDecisions;

    private List<double[]> learningDatataSetResults = new ArrayList<>();

    private double[] inputVector;
    private double lambda;
    private double learningRate;
    int correctResult=0;


    private double[][] weightsHiddenLayer;
    private double[][] weightsHiddenLayer2;
    private double[][] weightsHiddenLayer3;
    private double[][] weightsOutputLayer;


    private double[] biasHiddenLayer;
    private double[] biasHiddenLayer2;
    private double[] biasHiddenLayer3;
    private double[] biasOutputLayer;

    private double[] net1;
    private double[] Y1;
    private double[] net1_2;
    private double[] Y1_2;
    private double[] net1_3;
    private double[] Y1_3;
    private double[] net2;
    private double[] Y2;

    private double[] hiddenLayerError;
    private double[] hiddenLayerError2;
    private double[] hiddenLayerError3;
    private double[] outpuLayerError;


    public List<double[]> meanSquaredErrorBefore = new ArrayList<>();
    public List<double[]> meanSquaredErrorAfter = new ArrayList<>();


    NeuralNetwork3Layers(List<double[]> learningDataSetFeatures, List<Double> learningDataSetDecisions) {
        this.learningDataSetFeatures = learningDataSetFeatures;
        this.learningDataSetDecisions = learningDataSetDecisions;
    }


    public void calculateOutputForNetwork(double[] inputVector) {
        net1 = Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
        Y1 = Neuron.transformWithUnipolarSigmoidFunction(net1, lambda);

        net1_2 = Neuron.getOutputVector(weightsHiddenLayer2, Y1, biasHiddenLayer2);
        Y1_2 = Neuron.transformWithUnipolarSigmoidFunction(net1_2, lambda);


        net1_3 = Neuron.getOutputVector(weightsHiddenLayer3, Y1_2, biasHiddenLayer3);
        Y1_3 = Neuron.transformWithUnipolarSigmoidFunction(net1_3, lambda);

        net2 = Neuron.getOutputVector(weightsOutputLayer, Y1_3, biasOutputLayer);
        Y2 = Neuron.transformWithUnipolarStepFunction(net2);

        //for testing
        learningDatataSetResults.add(Y2);
    }



      public void calculateErrorsForLayers(double []expectedValuesOutputLayer ){
        outpuLayerError= new double[weightsOutputLayer.length];
        for(int i =0 ; i < outpuLayerError.length; i++){
            outpuLayerError[i]=Neuron.determineErrorFor0utputNeuron(expectedValuesOutputLayer[i],Y2[i] );
        }

        hiddenLayerError3= new double[weightsHiddenLayer3.length];
        for (int i =0 ; i < hiddenLayerError3.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsOutputLayer.length];//zmina z hidden3 na output layer
            for(int j=0 ; j< weightsForNextNeuron.length; j++){
                weightsForNextNeuron[j]=weightsOutputLayer[j][i]; // ?? ERROR
            }
            hiddenLayerError3[i]=Neuron.determineErrorForHiddenNeuron(outpuLayerError,weightsForNextNeuron,Y1_3[i], lambda);
        }


        hiddenLayerError2= new double[weightsHiddenLayer2.length];
        for (int i =0 ; i < hiddenLayerError2.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsHiddenLayer3.length];
            for(int j=0 ; j< weightsForNextNeuron.length; j++){
                weightsForNextNeuron[j]=weightsHiddenLayer3[j][i]; // ?? ERROR
            }
            hiddenLayerError2[i]=Neuron.determineErrorForHiddenNeuron(hiddenLayerError3,weightsForNextNeuron,Y1_2[i], lambda);
        }

        hiddenLayerError= new double[weightsHiddenLayer.length];
        for (int i =0 ; i < hiddenLayerError.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsHiddenLayer2.length];
            for(int j=0 ; j< weightsForNextNeuron.length; j++){
                weightsForNextNeuron[j]= weightsHiddenLayer2[j][i]; // ?? ERROR
            }
            hiddenLayerError[i]=Neuron.determineErrorForHiddenNeuron(hiddenLayerError2,weightsForNextNeuron,Y1[i], lambda);
        }

    }




    public double [] calculateNewBiasForHiddenLayer(double[] biasHiddenLayer,double [] hiddenLayerError){
        double[] newbiasHiddenLayer= new double[biasHiddenLayer.length];
        for (int i=0 ; i < newbiasHiddenLayer.length ; i ++){
            newbiasHiddenLayer[i]=Neuron.determineNewBiasForNeuron(biasHiddenLayer[i],learningRate,hiddenLayerError[i]);

        }
       // Matrix.showMatrix(newbiasHiddenLayer);
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
         //   Matrix.showMatrix(oldWeightsForOutputNeuron);

            double[] newWeightsForOutputNeuron= Neuron.determineWeightsForNeuron(oldWeightsForOutputNeuron,learningRate,outpuLayerError[i],Y1_3);
         //   System.out.println("new weights for neuron : ");
        ///    Matrix.showMatrix(newWeightsForOutputNeuron);

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
        //    System.out.println("old weights for neuron : ");
        //    Matrix.showMatrix(oldWeightsForHiddenNeuron);

            double[] newWeightsForHiddenNeuron= Neuron.determineWeightsForNeuron(oldWeightsForHiddenNeuron,learningRate,hiddenLayerError[i],inputVector);
         //   System.out.println("new weights for neuron : ");
        //    Matrix.showMatrix(newWeightsForHiddenNeuron);
//
            newWeightsForHiddenNeurons[i]=newWeightsForHiddenNeuron;
        }
        return  newWeightsForHiddenNeurons;
    }








    public void carryOutEpoch( double [] input,double [] expectedValuesOutputLayer){
        calculateOutputForNetwork(input);
        calculateErrorsForLayers(expectedValuesOutputLayer);


     /*   //calculateErrorsForLayers(weightsHiddenLayer2,weightsOutputLayer, expectedValuesOutputLayer);
     double [][]   newWeightsForHiddenNeurons3= new double[weightsHiddenLayer3.length][];
        for(int i=0 ; i< newWeightsForHiddenNeurons3.length ; i ++){

            double [] oldWeightsForHiddenNeuron= new double[weightsHiddenLayer3[i].length];

            for(int j=0; j<weightsHiddenLayer3[i].length; j++){
                oldWeightsForHiddenNeuron[j]=weightsHiddenLayer3[i][j];
            }

            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>old weights for neuron 3 layer : ");
            Matrix.showMatrix(oldWeightsForHiddenNeuron);

            //      double[] newWeightsForHiddenNeuron= determineWeightsForNeuron(oldWeightsForHiddenNeuron,learningRate,hiddenLayerError[i],inputVector);
            double newWeights[]= new double[oldWeightsForHiddenNeuron.length];
            for(int j =0 ; j < newWeights.length ; j++){
                newWeights[j]= learningRate *  hiddenLayerError3[i]  * Y1_2[j]; //error
            }
            for(int j =0 ; j < newWeights.length ; j++){
                newWeights[j]+=oldWeightsForHiddenNeuron[j];
            }

            newWeightsForHiddenNeurons3[i]=newWeights;
        }

        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>new weights for neuron 3 layer : ");
        Matrix.showMatrix(newWeightsForHiddenNeurons3);
           this.weightsHiddenLayer3=newWeightsForHiddenNeurons3;

     //   calculateWeightsForHiddenLayer(weightsHiddenLayer, learningRate, input);*/
  /*      System.out.println("------------------------------------------------------------------\n");
        calculateWeightsForOutputLayer();
        System.out.println("------------------------------------------------------------------\n");

        System.out.println("------------------------------------------------------------------\n ");
        System.out.println("\nweightsHiddenLayer3.length  AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"+weightsHiddenLayer3.length);
        Matrix.showMatrix(weightsHiddenLayer3);
        System.out.println("\n" +
                "weightsHiddenLayer3.length  AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"+hiddenLayerError3.length);

        Matrix.showMatrix(hiddenLayerError3);
       double [][] newWeightsForHiddenNeurons3= calculateWeightsForHiddenLayer(weightsHiddenLayer3,hiddenLayerError3,input);
        weightsHiddenLayer3=newWeightsForHiddenNeurons3;
        System.out.println("------------------------------------------------------------------\n ");
*/

     //   System.out.println("------------------------------------------------------------------\n ");
        double [][] newWeightsForHiddenNeurons3= calculateWeightsForHiddenLayer(weightsHiddenLayer3, hiddenLayerError3, Y1_2);
        weightsHiddenLayer3=newWeightsForHiddenNeurons3;
     //   System.out.println("------------------------------------------------------------------\n ");


     //   System.out.println("------------------------------------------------------------------\n ");
        double [][] newWeightsForHiddenNeurons2= calculateWeightsForHiddenLayer(weightsHiddenLayer2, hiddenLayerError2, Y1);
        weightsHiddenLayer2=newWeightsForHiddenNeurons2;
     //   System.out.println("------------------------------------------------------------------\n ");


       // System.out.println("------------------------------------------------------------------\nLAYER 3  BEFORE");
        double [][] newWeightsForHiddenNeurons= calculateWeightsForHiddenLayer(weightsHiddenLayer, hiddenLayerError, input);
        weightsHiddenLayer=newWeightsForHiddenNeurons;
     //   System.out.println("------------------------------------------------------------------\n LAYER AFTER");

        calculateWeightsForOutputLayer();


        double [] newBiasHiddenLayer=calculateNewBiasForHiddenLayer(biasHiddenLayer,hiddenLayerError);
        biasHiddenLayer=newBiasHiddenLayer;

        double [] newBiasHiddenLayer2=calculateNewBiasForHiddenLayer(biasHiddenLayer2,hiddenLayerError2);
        biasHiddenLayer2=newBiasHiddenLayer2;

        double [] newBiasHiddenLayer3=calculateNewBiasForHiddenLayer(biasHiddenLayer3,hiddenLayerError3);
        biasHiddenLayer3=newBiasHiddenLayer3;

        calculateNewBiasForOutputLayer();


    }






    public double calculateMeanSquaredError( ){
        double error=0;
        for(int i=0;i<learningDatataSetResults.size();i++){
        //  System.out.println(">>>>>>>>>>>>>>   expectedValues "+ learningDataSetDecisions.get(i) +"   >>>>>>>>>>  actualValue" +  learningDatataSetResults.get(i)[0]);
            error+=pow((learningDataSetDecisions.get(i)-learningDatataSetResults.get(i)[0]),2);
            if((learningDataSetDecisions.get(i)==learningDatataSetResults.get(i)[0])){
                correctResult++;
            }
           // System.out.println("error " + i + " " + error );
        }
        error= error/2;
     //   System.out.println(">>>>>>>>>>> ERROR " + error);
        return error;

    }















    public void setBiasOutputLayer(double[] biasOutputLayer) {
        this.biasOutputLayer = biasOutputLayer;
    }

    public void setBiasHiddenLayer3(double[] biasHiddenLayer3) {
        this.biasHiddenLayer3 = biasHiddenLayer3;
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

    public void setWeightsHiddenLayer3(double[][] weightsHiddenLayer3) {
        this.weightsHiddenLayer3 = weightsHiddenLayer3;
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

    public double[][] getWeightsHiddenLayer() {
        return weightsHiddenLayer;
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

    public double[] getBiasHiddenLayer3() {
        return biasHiddenLayer3;
    }





    public double[] getNet1() {
        return net1;
    }


    public double[] getNet1_2() {
        return net1_2;
    }


    public double[] getNet1_3() {
        return net1_3;
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

    public double[] getY1_3() {
        return Y1_3;
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
