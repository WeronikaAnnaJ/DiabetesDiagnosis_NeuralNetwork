package DiabetesDiagnosis;

import java.util.*;

import static java.lang.Math.pow;




public class NeuralNetwork {


    private List<double[]> learningDataSetFeatures;
    private List<Double> learningDataSetDecisions;
    private List<double[]> learningDatataSetResultsn=new ArrayList<>();


    private  List<String[]> testDataSet= new ArrayList<>();
    private List<double[]> testDataSetFeatures= new ArrayList<>();
    private List<Double> testDataSetDecisions= new ArrayList<>();


    private double [][] weightsHiddenLayer;
    private double [][] weightsHiddenLayer2;
    private double [][] weightsOutputLayer;

    private  double [] inputVector;
    private double lambda;
    int correctResult=0;

    private double[] biasHiddenLayer;
    private double[] biasHiddenLayer2;
    private double[] biasOutputLayer;

    private double [] net1;
    private double [] Y1;
    private double [] net1_2;
    private double [] Y1_2;
    private double [] net2;
    private double [] Y2;

    private double [] hiddenLayerError;
    private double [] hiddenLayerError2;
    private double [] outpuLayerError;

    private double [][] newWeightsForHiddenNeurons;
    private double [][] newWeightsForHiddenNeurons2;
    private double [][] newWeightsForOutputNeurons;
    private double learningRate;

    private double meanSquaredErrorBeforeEpoch;
    private double meanSquaredErrorAfterEpoch;

    public  List<Double>  meanSquaredErrors= new ArrayList<>();

    private List<double[]> testingDataSetFeatures;
    private List<Double> testingDataSetDecisions;

    private List<double[]> testingDatataSetResults = new ArrayList<>();
    private int correctTestingResults=0;




    NeuralNetwork(List<double[]> learningDataSetFeatures, List<Double> learningDataSetDecisions){
        this.learningDataSetFeatures=learningDataSetFeatures;
        this.learningDataSetDecisions=learningDataSetDecisions;
    }




    public void calculateOutputForNetwork(double [][] weightsHiddenLayer, double [][] weightsOutputLayer, double [] inputVector, double lambda){
        net1= Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
        Y1=Neuron.transformWithUnipolarSigmoidFunction(net1, lambda);
        net2 = Neuron.getOutputVector(weightsOutputLayer,Y1,biasOutputLayer);
        Y2 =Neuron.transformWithUnipolarSigmoidFunction(net2,lambda);
        learningDatataSetResultsn.add(Y2);
    }

    public void calculateOutputForNetwork(double [][] weightsHiddenLayer,double [][] weightsHiddenLayer2,  double [][] weightsOutputLayer, double [] inputVector, double lambda){
        net1= Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
        Y1=Neuron.transformWithUnipolarSigmoidFunction(net1, lambda);

        net1_2=Neuron.getOutputVector(weightsHiddenLayer2,Y1, biasHiddenLayer2);
        Y1_2=Neuron.transformWithUnipolarSigmoidFunction(net1_2, lambda);

        net2 = Neuron.getOutputVector(weightsOutputLayer,Y1_2,biasOutputLayer);
        Y2 =Neuron.transformWithUnipolarStepFunction(net2);

        learningDatataSetResultsn.add(Y2);
    }




    public void calculateErrorsForLayers( double[][] weightsHiddenLayer,double[][] weightsOutputLayer, double []expectedValuesOutputLayer ){
        outpuLayerError= new double[weightsOutputLayer.length];
        for(int i =0 ; i < outpuLayerError.length; i++){
            outpuLayerError[i]=Neuron.determineErrorFor0utputNeuron(expectedValuesOutputLayer[i],Y2[i], lambda, Y2[i] );
        }

        hiddenLayerError= new double[weightsHiddenLayer.length];
        for (int i =0 ; i < hiddenLayerError.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsOutputLayer.length];
            for(int j=0 ; j< weightsOutputLayer.length; j++){
                weightsForNextNeuron[j]=weightsOutputLayer[j][i]; // ?? ERROR
            }
            hiddenLayerError[i]=Neuron.determineErrorForHiddenNeuron(outpuLayerError,weightsForNextNeuron,Y1[i], lambda);
        }

        hiddenLayerError= new double[weightsHiddenLayer.length];
        for (int i =0 ; i < hiddenLayerError.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsOutputLayer.length];
            for(int j=0 ; j< weightsOutputLayer.length; j++){
                weightsForNextNeuron[j]=weightsOutputLayer[j][i]; // ?? ERROR
            }
            hiddenLayerError[i]=Neuron.determineErrorForHiddenNeuron(outpuLayerError,weightsForNextNeuron,Y1[i], lambda);
        }
    }



    public void calculateErrorsForLayers( double[][] weightsHiddenLayer,double[][] weightsHiddenLayer2, double[][] weightsOutputLayer, double []expectedValuesOutputLayer ){
        outpuLayerError= new double[weightsOutputLayer.length];
        for(int i =0 ; i < outpuLayerError.length; i++){
            outpuLayerError[i]=Neuron.determineErrorFor0utputNeuron(expectedValuesOutputLayer[i],Y2[i] );
        }

        hiddenLayerError2= new double[weightsHiddenLayer2.length];
        for (int i =0 ; i < hiddenLayerError2.length ; i ++){
            double [] weightsForNextNeuron= new double[hiddenLayerError2.length];
            for(int j=0 ; j< weightsOutputLayer.length; j++){
                weightsForNextNeuron[j]=weightsOutputLayer[j][i];
            }
            hiddenLayerError2[i]=Neuron.determineErrorForHiddenNeuron(outpuLayerError,weightsForNextNeuron,Y1_2[i], lambda);
        }

        hiddenLayerError= new double[weightsHiddenLayer.length];
        for (int i =0 ; i < hiddenLayerError.length ; i ++){
            double [] weightsForNextNeuron= new double[hiddenLayerError2.length];
            for(int j=0 ; j< hiddenLayerError2.length; j++){
                weightsForNextNeuron[j]= weightsHiddenLayer2[j][i];
            }
            hiddenLayerError[i]=Neuron.determineErrorForHiddenNeuron(hiddenLayerError2,weightsForNextNeuron,Y1[i], lambda);
        }
    }




    public void calculateWeightsForHiddenLayer( double[][] weightsHiddenLayer, double learningRate,double [] inputVector){
        newWeightsForHiddenNeurons= new double[weightsHiddenLayer.length][];

        for(int i=0 ; i< newWeightsForHiddenNeurons.length ; i ++){
            double [] oldWeightsForHiddenNeuron= new double[weightsHiddenLayer[i].length];
            for(int j=0; j<weightsHiddenLayer[i].length; j++){
                oldWeightsForHiddenNeuron[j]=weightsHiddenLayer[i][j];
            }
            double[] newWeightsForHiddenNeuron= Neuron.determineWeightsForNeuron(oldWeightsForHiddenNeuron,learningRate,hiddenLayerError[i],inputVector);
            newWeightsForHiddenNeurons[i]=newWeightsForHiddenNeuron;
        }
        this.weightsHiddenLayer=newWeightsForHiddenNeurons;
    }




    public void calculateWeightsForHiddenLayer( double[][] weightsHiddenLayer, double learningRate,double [] inputVector, int layerNumber){
        double [][] newWeightsForHiddenNeurons= new double[weightsHiddenLayer.length][];
        for(int i=0 ; i< newWeightsForHiddenNeurons.length ; i ++){
            double [] oldWeightsForHiddenNeuron= new double[weightsHiddenLayer[i].length];

            for(int j=0; j<weightsHiddenLayer[i].length; j++){
                oldWeightsForHiddenNeuron[j]=weightsHiddenLayer[i][j];
            }
            double[] newWeightsForHiddenNeuron= Neuron.determineWeightsForNeuron(oldWeightsForHiddenNeuron,learningRate,hiddenLayerError[i],inputVector);
            newWeightsForHiddenNeurons[i]=newWeightsForHiddenNeuron;
        }
        if( layerNumber==1){
            this.weightsHiddenLayer=newWeightsForHiddenNeurons;
        }else{
            this.weightsHiddenLayer2=newWeightsForHiddenNeurons;
        }
    }




    public void calculateNewBiasForHiddenLayer(double[] biasHiddenLayer,double learningRate,double [] hiddenLayerError){
        double[] newbiasHiddenLayer= new double[biasHiddenLayer.length];
        for (int i=0 ; i < newbiasHiddenLayer.length ; i ++){
            newbiasHiddenLayer[i]=Neuron.determineNewBiasForNeuron(biasHiddenLayer[i],learningRate,hiddenLayerError[i]);
        }
        this.biasHiddenLayer=newbiasHiddenLayer;
    }



    public void calculateNewBiasForHiddenLayer(double[] biasHiddenLayer,double learningRate,double [] hiddenLayerError, int layerNumber){
        double[] newbiasHiddenLayer= new double[biasHiddenLayer.length];
        for (int i=0 ; i < newbiasHiddenLayer.length ; i ++){
            newbiasHiddenLayer[i]=Neuron.determineNewBiasForNeuron(biasHiddenLayer[i],learningRate,hiddenLayerError[i]);
        }
        if(layerNumber==1){
            this.biasHiddenLayer=newbiasHiddenLayer;
        }else{
            this.biasHiddenLayer2=newbiasHiddenLayer;
        }
    }



    public void calculateWeightsForOutputLayer(double[][] weightsOutputLayer, double learningRate,double [] Y1){
        newWeightsForOutputNeurons= new double[weightsOutputLayer.length][];
        for(int i=0 ; i< newWeightsForOutputNeurons.length ; i ++){
            double [] oldWeightsForOutputNeuron= new double[weightsOutputLayer[i].length];
            for(int j=0; j<weightsOutputLayer[i].length; j++){
                oldWeightsForOutputNeuron[j]=weightsOutputLayer[i][j];
            }
            double[] newWeightsForOutputNeuron= Neuron.determineWeightsForNeuron(oldWeightsForOutputNeuron,learningRate,outpuLayerError[i],Y1);
            newWeightsForOutputNeurons[i]=newWeightsForOutputNeuron;
        }
        this.weightsOutputLayer=newWeightsForOutputNeurons;
    }



    public void calculateNewBiasForOutputLayer(double[] biasOutputLayer,double learningRate,double [] outpuLayerError){
        double[] newBiasOutputLayer= new double[biasOutputLayer.length];
        for (int i=0 ; i < newBiasOutputLayer.length ; i ++){
            newBiasOutputLayer[i]=Neuron.determineNewBiasForNeuron(biasOutputLayer[i],1,outpuLayerError[i]);
        }
        this.biasOutputLayer=newBiasOutputLayer;
    }



    public void carryOutEpoch( double [] input,double [] expectedValuesOutputLayer, double[] biasHiddenLayer , double[] biasOutputLayer){
        calculateOutputForNetwork(weightsHiddenLayer,weightsOutputLayer,input,lambda);
        this.meanSquaredErrorBeforeEpoch=calculateMeanSquaredError(expectedValuesOutputLayer,Y2);
        learningDatataSetResultsn.add(Y2);
        calculateMeanSquaredError(expectedValuesOutputLayer,Y2);

        calculateErrorsForLayers(weightsHiddenLayer,weightsOutputLayer, expectedValuesOutputLayer);
        calculateWeightsForHiddenLayer(weightsHiddenLayer,learningRate,input);
        calculateNewBiasForHiddenLayer(biasHiddenLayer,learningRate, hiddenLayerError);//decide where put values matrix, atripute ?
        calculateWeightsForOutputLayer(weightsOutputLayer,learningRate,Y1);

        calculateNewBiasForOutputLayer(biasOutputLayer,learningRate, outpuLayerError);

        calculateOutputForNetwork(this.newWeightsForHiddenNeurons,this.newWeightsForOutputNeurons,input,lambda);
        this.meanSquaredErrorAfterEpoch=calculateMeanSquaredError(expectedValuesOutputLayer,Y2);

    }



    public void carryOutEpoch( double [] input,double [] expectedValuesOutputLayer, double[] biasHiddenLayer, double[] biasHiddenLayer2, double[] biasOutputLayer){
        calculateOutputForNetwork(weightsHiddenLayer,weightsHiddenLayer2, weightsOutputLayer,input,lambda);
        this.meanSquaredErrorBeforeEpoch=calculateMeanSquaredError(expectedValuesOutputLayer,Y2);

        calculateErrorsForLayers(weightsHiddenLayer,weightsHiddenLayer2, weightsOutputLayer, expectedValuesOutputLayer);
        newWeightsForHiddenNeurons2= new double[weightsHiddenLayer2.length][];
        for(int i=0 ; i< newWeightsForHiddenNeurons2.length ; i ++){
            double [] oldWeightsForHiddenNeuron= new double[weightsHiddenLayer2[i].length];
            for(int j=0; j<weightsHiddenLayer2[i].length; j++){
                oldWeightsForHiddenNeuron[j]=weightsHiddenLayer2[i][j];
            }
            double newWeights[]= new double[oldWeightsForHiddenNeuron.length];
            for(int j =0 ; j < newWeights.length ; j++){
                newWeights[j]= learningRate *  hiddenLayerError2[i]  * Y1[j]; //error
            }
            for(int j =0 ; j < oldWeightsForHiddenNeuron.length ; j++){
                newWeights[j]+=oldWeightsForHiddenNeuron[j];
            }
            newWeightsForHiddenNeurons2[i]=newWeights;
        }
        this.weightsHiddenLayer2=newWeightsForHiddenNeurons2;

        calculateWeightsForHiddenLayer(weightsHiddenLayer, learningRate, input);
        calculateWeightsForOutputLayer(weightsOutputLayer,learningRate,Y1_2);

        calculateNewBiasForHiddenLayer(biasHiddenLayer,learningRate, hiddenLayerError,1);//decide where put values matrix, atripute ?
        calculateNewBiasForHiddenLayer(biasHiddenLayer2,learningRate, hiddenLayerError2,2);
        calculateNewBiasForOutputLayer(biasOutputLayer,learningRate, outpuLayerError);


    }



    public double calculateMeanSquaredError(double [] expectedValues,  double [] actualValues ){
         double error=0.0;
        for(int i=0;i<expectedValues.length;i++){
            error+=pow((expectedValues[i]-actualValues[i]),2);
        }
        error= error/2;
        meanSquaredErrors.add(error);
        return error;

    }



    public double calculateMeanSquaredError( ){
        double error=0.0;
        for(int i=0;i<learningDatataSetResultsn.size();i++){
            error+=pow((learningDataSetDecisions.get(i)-learningDatataSetResultsn.get(i)[0]),2);
            if(learningDataSetDecisions.get(i)==learningDatataSetResultsn.get(i)[0]){
                correctResult++;
            }
        }
        error= error/2;
        return error;
    }



    public double testAccurancy(List<double[]> testingDataSetFeatures, List<Double> testingDataSetDecisions) {
        this.testingDataSetFeatures = testingDataSetFeatures;
        this.testingDataSetDecisions = testingDataSetDecisions;
        resetCorrectTestingResults();
        testingDatataSetResults.removeAll(testingDatataSetResults);
        for(int i =0 ; i < testingDataSetFeatures.size() ;i ++){
            inputVector=testingDataSetFeatures.get(i);

            net1= Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
            Y1=Neuron.transformWithUnipolarSigmoidFunction(net1, lambda);

            net1_2=Neuron.getOutputVector(weightsHiddenLayer2,Y1, biasHiddenLayer2);
            Y1_2=Neuron.transformWithUnipolarSigmoidFunction(net1_2, lambda);

            net2 = Neuron.getOutputVector(weightsOutputLayer,Y1_2,biasOutputLayer);
            Y2 =Neuron.transformWithUnipolarStepFunction(net2);

            testingDatataSetResults.add(Y2);
        }
        for(int i=0;i<testingDatataSetResults.size();i++){
            if(testingDataSetDecisions.get(i)==testingDatataSetResults.get(i)[0]){
                correctTestingResults++;
            }
        }
        int allResults= testingDatataSetResults.size();
        int correctResults= correctTestingResults;

        System.out.println(correctResults + "/" + allResults);
        double correctInPercentages= (correctResults * 100 )/allResults;
        System.out.println(correctInPercentages + " %");
        return  correctInPercentages;
    }



    public int getCorrectTestingResults() {
        return correctTestingResults;
    }
    public void  resetCorrectTestingResults() {
        correctTestingResults=0;
    }



    public double[] getNet1() {
        return net1;
    }

    public double[] getNet2() {
        return net2;
    }

    public double[] getNet1_2() {
        return net1_2;
    }

    public double[] getY1() {
        return Y1;
    }

    public double[] getY1_2() {
        return Y1_2;
    }

    public double[] getY2() {
        return Y2;
    }

    public double[] getHiddenLayerError() {
        return hiddenLayerError;
    }

    public double[] getOutpuLayerError() {
        return outpuLayerError;
    }

    public double[][] getWeightsForOutputLayer() {
        return weightsOutputLayer;
    }

    public double[][] getWeightsForHiddenLayer() {
        return weightsHiddenLayer;
    }


    public double[][] getWeightsForHiddenLayer2() {
        return weightsHiddenLayer2;
    }


    public double[] getBiasHiddenLayer() {
        return biasHiddenLayer;
    }

    public double[] getBiasOutputLayer() {
        return biasOutputLayer;
    }



    public List<double[]> getLearningDataSetFeatures() {
        return learningDataSetFeatures;
    }

    public List<Double> getLearningDataSetDecisions() {
        return learningDataSetDecisions;
    }

    public double getMeanSquaredErrorAfterEpoch() {
        return meanSquaredErrorAfterEpoch;
    }

    public double getMeanSquaredErrorBeforeEpoch() {
        return meanSquaredErrorBeforeEpoch;
    }


    public List<double[]> getLearningDatataSetResults() {
        return learningDatataSetResultsn;
    }

    public void setBiasOutputLayer(double[] biasOutputLayer) {
        this.biasOutputLayer = biasOutputLayer;
    }

    public double[] getBiasHiddenLayer2() {
        return biasHiddenLayer2;
    }

    public void setBiasHiddenLayer(double[] biasHiddenLayer) {
        this.biasHiddenLayer = biasHiddenLayer;
    }

    public void setBiasHiddenLayer2(double[] biasHiddenLayer2) {
        this.biasHiddenLayer2 = biasHiddenLayer2;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }


    public void setWeightsHiddenLayer(double[][] weightsHiddenLayer) {
        this.weightsHiddenLayer = weightsHiddenLayer;
    }


    public void setWeightsHiddenLayer2(double[][] weightsHiddenLayer2) {
        this.weightsHiddenLayer2 = weightsHiddenLayer2;
    }


    public void setWeightsOutputLayer(double[][] weightsOutputLayer) {
        this.weightsOutputLayer = weightsOutputLayer;
    }



    public void setNewWeightsForHiddenNeurons(double[][] newWeightsForHiddenNeurons) {
        this.weightsHiddenLayer= newWeightsForHiddenNeurons;
    }

    public void setNewWeightsForOutputNeurons(double[][] newWeightsForOutputNeurons) {
        this.weightsOutputLayer= newWeightsForOutputNeurons;
    }


    public void resetLearningDataSetResults() {
        learningDatataSetResultsn.removeAll(learningDatataSetResultsn);
    }

    public void resetMeanSquaredErrors() {
        meanSquaredErrors.removeAll(meanSquaredErrors);
    }


    public List<Double> getMeanSquaredErrors() {
        return meanSquaredErrors;
    }

    public void resetCorrectResults(){
        correctResult=0;
    }

    public int getCorrectResult() {
        return correctResult;
    }

}
