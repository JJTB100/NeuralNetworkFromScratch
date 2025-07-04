package neural.networks;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Layer {
    private String name;
    private int inputSize;
    private int outputSize;
    private INDArray weights;
    private INDArray biases;
    private INDArray weightsAddBiases;
    private INDArray activatedOutput;
    private INDArray dWeights;
    private INDArray dBiases;
    private INDArray dActivatedOutput;

    public Layer(String name, int outputSize, int inputSize) {
        this.name = name;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public String getName() {
        return name;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public INDArray getWeights() {
        return weights;
    }

    public void setWeights(INDArray weights) {
        this.weights = weights;
    }

    public INDArray getBiases() {
        return biases;
    }

    public void setBiases(INDArray biases) {
        this.biases = biases;
    }

    public void initializeWeights() {
        double scale = Math.sqrt(2.0 / inputSize);
        this.weights = Nd4j.randn(outputSize, inputSize).mul(scale);

    }

    public void initializeBiases() {
        this.biases = Nd4j.zeros(outputSize, 1);
    }

    public INDArray getWeightsAddBiases() {
        return weightsAddBiases;
    }

    public void setWeightsAddBiases(INDArray weightsAddBiases) {
        this.weightsAddBiases = weightsAddBiases;
    }

    public INDArray getActivatedOutput() {
        return activatedOutput;
    }

    public void setActivatedOutput(INDArray activatedOutput) {
        this.activatedOutput = activatedOutput;
    }

    public INDArray getDWeights() {
        return dWeights;
    }

    public void setDWeights(INDArray dWeights) {
        this.dWeights = dWeights;
    }

    public INDArray getDBiases() {
        return dBiases;
    }

    public void setDBiases(INDArray dBiases) {
        this.dBiases = dBiases;
    }

    public INDArray getDActivatedOutput() {
        return dActivatedOutput;
    }

    public void setDActivatedOutput(INDArray dActivatedOutput) {
        this.dActivatedOutput = dActivatedOutput;
    }

}
