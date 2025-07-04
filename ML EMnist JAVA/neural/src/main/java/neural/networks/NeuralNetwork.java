package neural.networks;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax;
import org.nd4j.linalg.factory.Nd4j;

public class NeuralNetwork {
    private List<Layer> layers;
    private int numberClasses;
    private String FILENAME;
    private String testFILENAME;
    private int inputSize;
    private int epochs;
    private double alpha;
    private int batchSize;

    public NeuralNetwork(int inputSize, int numberClasses, String csvFILENAME, String testFILENAME,
            List<Integer> HiddenNeuralNetworkSize,
            int epochs, double alpha, int batchSize) {
        if (HiddenNeuralNetworkSize.size() == 0) {
            throw new IllegalArgumentException("Layer count must be greater than 0");
        }
        this.inputSize = inputSize;
        this.epochs = epochs;
        this.alpha = alpha;
        this.batchSize = batchSize;
        this.FILENAME = csvFILENAME;
        this.testFILENAME = testFILENAME;
        layers = new ArrayList<>();
        this.numberClasses = numberClasses;

        for (int i = 0; i < HiddenNeuralNetworkSize.size(); i++) {
            int input = (i == 0) ? inputSize : HiddenNeuralNetworkSize.get(i - 1);
            int output = HiddenNeuralNetworkSize.get(i);
            layers.add(new Layer("Hidden Layer " + (i + 1), output, input));
        }
        layers.add(new Layer("Output Layer", numberClasses,
                HiddenNeuralNetworkSize.get(HiddenNeuralNetworkSize.size() - 1)));

        for (Layer layer : layers) {
            layer.initializeWeights();
            layer.initializeBiases();
        }

    }

    public void forward_propagation(INDArray X) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            INDArray Z;

            if (i == 0) {
                // First hidden layer: input X
                Z = layer.getWeights().mmul(X).addColumnVector(layer.getBiases());
            } else {
                Layer prevLayer = layers.get(i - 1);
                Z = layer.getWeights().mmul(prevLayer.getActivatedOutput()).addColumnVector(layer.getBiases());
            }

            layer.setWeightsAddBiases(Z);

            if (i == layers.size() - 1) {
                // Output layer: softmax
                INDArray output = Nd4j.zeros(Z.shape());
                SoftMax sm = new SoftMax(Z, output, 0); // Assuming axis 0 (batch)
                Nd4j.getExecutioner().exec(sm);
                layer.setActivatedOutput(output);
            } else {
                layer.setActivatedOutput(NMaths.leakyReLU(Z));
            }
        }

    }

    private INDArray one_hot(INDArray Y) {
        long m = Y.length(); // number of samples
        INDArray oneHot = Nd4j.zeros(numberClasses, m);
        for (int i = 0; i < m; i++) {
            int classLabel = Y.getInt(i);
            oneHot.putScalar(classLabel, i, 1.0);
        }
        return oneHot;
    }

    public void backward_propagation(INDArray X, INDArray Y, double alpha) {
        long m = X.shape()[1];
        INDArray Y_onehot = one_hot(Y);

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            INDArray output = layer.getActivatedOutput();

            if (i == layers.size() - 1) {
                // Output layer
                layer.setDActivatedOutput(output.sub(Y_onehot));
            } else {
                Layer nextLayer = layers.get(i + 1);
                INDArray delta = nextLayer.getWeights().transpose().mmul(nextLayer.getDActivatedOutput())
                        .mul(NMaths.derivativeLeakyReLU(layer.getWeightsAddBiases()));
                layer.setDActivatedOutput(delta);
            }

            INDArray prevActivation = (i == 0) ? X : layers.get(i - 1).getActivatedOutput();
            layer.setDWeights(layer.getDActivatedOutput().mmul(prevActivation.transpose()).div(m));
            layer.setDBiases(layer.getDActivatedOutput().sum(1).div(m));
        }

        for (Layer layer : layers) {
            layer.setWeights(layer.getWeights().sub(layer.getDWeights().mul(alpha)));
            layer.setBiases(layer.getBiases().sub(layer.getDBiases().mul(alpha)));
        }
    }

    public void train() {
        int totalLines = 0;
        try (BufferedReader lineCounter = new BufferedReader(new FileReader(FILENAME))) {
            while (lineCounter.readLine() != null) {
                totalLines++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Total lines in file: " + totalLines);
        long lastTime = System.currentTimeMillis();

        for (int epoch = 0; epoch < epochs; epoch++) {
            try (BufferedReader br = new BufferedReader(new FileReader(FILENAME))) {
                String line;
                float[][] batch = new float[batchSize][inputSize];
                int lineCount = 0;
                int nextFreeBatchLine = 0;

                while ((line = br.readLine()) != null) {
                    if (lineCount == 0) {
                        lineCount++;
                        continue;
                    }
                    String[] strValues = line.split(",");
                    float[] values = new float[strValues.length];
                    for (int i = 0; i < strValues.length; i++) {
                        values[i] = Float.parseFloat(strValues[i]);
                    }

                    batch[nextFreeBatchLine] = values;
                    if (lineCount % 1000 == 0) {
                        displayProgress(epoch, lineCount, totalLines, System.currentTimeMillis()-lastTime);
                        lastTime = System.currentTimeMillis();
                    }
                    lineCount++;
                    nextFreeBatchLine++;

                    if (lineCount % batchSize == 0) {
                        // Process the current batch
                        processBatch(batch);
                        batch = new float[batchSize][inputSize]; // clear the batch for the next lines
                        nextFreeBatchLine = 0;
                    }
                }
                displayProgress(epoch, totalLines, totalLines, System.currentTimeMillis() - lastTime);

                // Process any remaining lines if file lines not divisible by batchSize
                if (nextFreeBatchLine > 0) {
                    float[][] lastBatch = new float[inputSize][nextFreeBatchLine];
                    for (int i = 0; i < inputSize; i++) {
                        System.arraycopy(batch[i], 0, lastBatch[i], 0, nextFreeBatchLine);
                    }
                    processBatch(lastBatch);

                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        testNN(testFILENAME);

    }

    private void displayProgress(int epoch, int lineCount, int totalLines, long timeSinceLast) {
        int progressBarLength = 50;
        double progress = (double) lineCount / totalLines;
        int filledLength = (int) Math.ceil(progressBarLength * progress);
        StringBuilder bar = new StringBuilder();
        bar.append("[");
        for (int i = 0; i < progressBarLength; i++) {
            if (i < filledLength) {
                bar.append("=");
            } else {
                bar.append(" ");
            }
        }
        bar.append("]");
        long timeLeft = (((epochs-epoch) * totalLines) + (totalLines-lineCount)/1000 * timeSinceLast)/1000;
        System.out.print("\rCurrent Epoch: " + String.format("%2d", epoch) + ". " 
            + String.format("%6d", lineCount) + "/" + String.format("%6d", totalLines)
            + " samples processed. Time for last 1000 samples: " + String.format("%5d", timeSinceLast) + "ms. Predicted Total Time Left: " 
            + String.format("%5d", timeLeft) + "s. "
            + bar.toString());
        System.out.flush();
    }

    private void processBatch(float[][] data) {
        int m = data.length; // number of samples in the batch
        int nFeaturesPlusLabel = data[0].length; // number of columns: label + features
        float[][] features = new float[m][nFeaturesPlusLabel - 1];
        int[] labels = new int[m];
        for (int i = 0; i < m; i++) {
            labels[i] = (int) data[i][0]; // first column is label
            for (int j = 1; j < nFeaturesPlusLabel - 1; j++) {
                features[i][j - 1] = data[i][j];
            }
        }
        // Convert to INDArrays
        INDArray X = Nd4j.create(features).transpose(); // shape [784, m]
        INDArray Y = Nd4j.createFromArray(labels); // shape [m]
        // Normalize features
        X = X.div((float) 255.0);

        forward_propagation(X);
        backward_propagation(X, Y, alpha);

    }

    public INDArray predict(INDArray X) {
        forward_propagation(X);
        Layer outputLayer = layers.get(layers.size() - 1);
        return outputLayer.getActivatedOutput().argMax(0); // Returns the index of the maximum value in each row

    }

    public float testNN(String FILENAME) {
        List<float[]> dataList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(FILENAME))) {
            String line;
            boolean isFirst = true;
            while ((line = br.readLine()) != null) {
                if (isFirst) {
                    isFirst = false;
                    continue;
                }
                String[] strValues = line.split(",");
                float[] features = new float[strValues.length - 1];
                for (int i = 1; i < strValues.length; i++) {
                    features[i - 1] = Float.parseFloat(strValues[i]);
                }
                dataList.add(features);
                labelList.add(Integer.parseInt(strValues[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
            return 0f;
        }
        int m = dataList.size();
        int n = dataList.get(0).length;
        float[][] featureArr = new float[m][n];
        int[] labelArr = new int[m];
        for (int i = 0; i < m; i++) {
            featureArr[i] = dataList.get(i);
            labelArr[i] = labelList.get(i);
        }
        INDArray X = Nd4j.create(featureArr).transpose(); // shape [n, m]
        X = X.div(255.0f);
        INDArray Y = Nd4j.createFromArray(labelArr);
        INDArray predictions = predict(X);
        long correct = 0;
        for (int i = 0; i < m; i++) {
            if (predictions.getInt(i) == Y.getInt(i))
                correct++;
        }
        System.out.println("\nAccuracy: " + (double) correct / m);
        return (float) correct / m;
    }

}
