package DiabetesDiagnosis;



import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.pow;

public class NeutalNetwork1Layer {

    private List<double[]> learningDataSetFeatures;
    private List<Double> learningDataSetDecisions;
    private List<double[]> learningDatataSetResults = new ArrayList<>();


    private List<double[]> testingDataSetFeatures;
    private List<Double> testingDataSetDecisions;
    private List<double[]> testingDatataSetResults = new ArrayList<>();


    private double[] inputVector;
    private double lambda;
    private double learningRate;
    private int correctResult=0;

    private double[][] weightsHiddenLayer;
    private double[][] weightsOutputLayer;


    private double[] biasHiddenLayer;
    private double[] biasOutputLayer;

    private double[] net1;
    private double[] Y1;
    private double[] net2;
    private double[] Y2;

    private double[] hiddenLayerError;
    private double[] outpuLayerError;

    private List<Double> meanSquaredErrors= new ArrayList<>();

    private int correctTestingResults=0;




    NeutalNetwork1Layer(List<double[]> learningDataSetFeatures, List<Double> learningDataSetDecisions) {
        this.learningDataSetFeatures = learningDataSetFeatures;
        this.learningDataSetDecisions = learningDataSetDecisions;
    }


    public void calculateOutputForNetwork(double[] inputVector) {
        net1 = Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
        Y1 = Neuron.transformWithUnipolarSigmoidFunction(net1, lambda);

        net2 = Neuron.getOutputVector(weightsOutputLayer, Y1, biasOutputLayer);
        Y2 = Neuron.transformWithUnipolarStepFunction(net2);

        learningDatataSetResults.add(Y2);

    }



    public void calculateErrorsForLayers(double []expectedValuesOutputLayer ){
        outpuLayerError= new double[weightsOutputLayer.length];
        for(int i =0 ; i < outpuLayerError.length; i++){
            outpuLayerError[i]=Neuron.determineErrorFor0utputNeuron(expectedValuesOutputLayer[i],Y2[i] );
        }
        hiddenLayerError= new double[weightsHiddenLayer.length];
        for (int i =0 ; i < hiddenLayerError.length ; i ++){
            double [] weightsForNextNeuron= new double[weightsOutputLayer.length];
            for(int j=0 ; j< weightsForNextNeuron.length; j++){
                weightsForNextNeuron[j]= weightsOutputLayer[j][i]; // ?? ERROR
            }
            hiddenLayerError[i]=Neuron.determineErrorForHiddenNeuron(outpuLayerError,weightsForNextNeuron,Y1[i], lambda);
        }
    }



    public double [] calculateNewBiasForHiddenLayer(double[] biasHiddenLayer,double [] hiddenLayerError){
        double[] newbiasHiddenLayer= new double[biasHiddenLayer.length];
        for (int i=0 ; i < newbiasHiddenLayer.length ; i ++){
            newbiasHiddenLayer[i]=Neuron.determineNewBiasForNeuron(biasHiddenLayer[i],learningRate,hiddenLayerError[i]);
        }
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
            double[] newWeightsForOutputNeuron= Neuron.determineWeightsForNeuron(oldWeightsForOutputNeuron,learningRate,outpuLayerError[i],Y1);
            newWeightsForOutputNeurons[i]=newWeightsForOutputNeuron;
        }
        this.weightsOutputLayer=newWeightsForOutputNeurons;
    }



    public double [][] calculateWeightsForHiddenLayer( double[][] weightsHiddenLayer, double[] hiddenLayerError, double [] inputVector){
        double [][] newWeightsForHiddenNeurons= new double[weightsHiddenLayer.length][];

        for(int i=0 ; i< newWeightsForHiddenNeurons.length ; i ++){
            double [] oldWeightsForHiddenNeuron= new double[weightsHiddenLayer[i].length];
            for(int j=0; j<weightsHiddenLayer[i].length; j++){
                oldWeightsForHiddenNeuron[j]=weightsHiddenLayer[i][j];
            }
            double[] newWeightsForHiddenNeuron= Neuron.determineWeightsForNeuron(oldWeightsForHiddenNeuron,learningRate,hiddenLayerError[i],inputVector);
            newWeightsForHiddenNeurons[i]=newWeightsForHiddenNeuron;
        }
        return  newWeightsForHiddenNeurons;
    }



    public void carryOutEpoch( double [] input,double [] expectedValuesOutputLayer){
        calculateOutputForNetwork(input);
        calculateErrorsForLayers(expectedValuesOutputLayer);
        double [][] newWeightsForHiddenNeurons= calculateWeightsForHiddenLayer(weightsHiddenLayer, hiddenLayerError, input);
        weightsHiddenLayer=newWeightsForHiddenNeurons;
        calculateWeightsForOutputLayer();
        double [] newBiasHiddenLayer=calculateNewBiasForHiddenLayer(biasHiddenLayer,hiddenLayerError);
        biasHiddenLayer=newBiasHiddenLayer;
        calculateNewBiasForOutputLayer();


    }



    public double calculateMeanSquaredError(){
        double error=0;
        for(int i=0;i<learningDatataSetResults.size();i++){
            error+=pow((learningDataSetDecisions.get(i)-learningDatataSetResults.get(i)[0]),2);
            if(learningDataSetDecisions.get(i)==learningDatataSetResults.get(i)[0]){
                correctResult++;
            }
        }
        error= error/2;
        return error;
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



    public double testAccurancy(List<double[]> testingDataSetFeatures, List<Double> testingDataSetDecisions) {
        this.testingDataSetFeatures = testingDataSetFeatures;
        this.testingDataSetDecisions = testingDataSetDecisions;
        resetCorrectTestingResults();
        testingDatataSetResults.removeAll(testingDatataSetResults);
        for(int i =0 ; i < testingDataSetFeatures.size() ;i ++){
            inputVector=testingDataSetFeatures.get(i);
            net1 = Neuron.getOutputVector(weightsHiddenLayer, inputVector, biasHiddenLayer);
            Y1 = Neuron.transformWithUnipolarSigmoidFunction(net1, lambda);
            net2 = Neuron.getOutputVector(weightsOutputLayer, Y1, biasOutputLayer);
            Y2 = Neuron.transformWithUnipolarStepFunction(net2);
            testingDatataSetResults.add(Y2);
        }

        for(int i=0;i<testingDatataSetResults.size();i++){
            if(testingDataSetDecisions.get(i)==testingDatataSetResults.get(i)[0]) correctTestingResults++;
        }
        int allResults= testingDatataSetResults.size();
        int correctResults= correctTestingResults;
        System.out.println(correctResults + "/" + allResults);
        double correctInPercentages= (correctResults * 100 )/allResults;
        System.out.println(correctInPercentages + " %");
        return  correctInPercentages;
    }


    public void setBiasOutputLayer(double[] biasOutputLayer) {
        this.biasOutputLayer = biasOutputLayer;
    }

    public void setBiasHiddenLayer(double[] biasHiddenLayer) {
        this.biasHiddenLayer = biasHiddenLayer;
    }


    public void setWeightsOutputLayer(double[][] weightsOutputLayer) {
        this.weightsOutputLayer = weightsOutputLayer;
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


    public double[] getNet1() {
        return net1;
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


    public double[] getHiddenLayerError() {
        return hiddenLayerError;
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

    public List<Double> getMeanSquaredErrors() {
        return meanSquaredErrors;
    }

    public void  resetMeanSquaredErrors() {
        meanSquaredErrors.removeAll(meanSquaredErrors);
    }

    public void  setDataSets(List<double []> multipliedLearningDataSetFeatures, List<Double> multipliedLearningDataSetDecisions){
        this.learningDataSetDecisions= multipliedLearningDataSetDecisions;
        this.learningDataSetFeatures= multipliedLearningDataSetFeatures;
    }


    public void resetCorrectResults(){
        correctResult=0;
    }

    public int getCorrectResult() {
        return correctResult;
    }

    public int getCorrectTestingResults() {
        return correctTestingResults;
    }
    public void  resetCorrectTestingResults() {
        correctTestingResults=0;
    }

    public void resetlearningDataSets(){
        this.learningDataSetFeatures.removeAll(learningDataSetFeatures);
        this.learningDataSetDecisions.removeAll(learningDataSetDecisions);
    }

}
