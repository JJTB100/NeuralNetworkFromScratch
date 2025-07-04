package neural.networks;

import java.util.List;

// import onnx.OnnxMlProto3.TensorProto.DataType;

public class Main {
    public static void main(String[] args) {
        int numberOfClasses = 10;
        String FILENAME = "neural/src/main/resources/emnist-digits-train.csv";
        String testFILENAME = "neural/src/main/resources/emnist-digits-test.csv";
        List<Integer> HiddenLayerNumNeurons = List.of(300, 100, 30);
        int epochs = 10;
        int batchSize = 100;
        double learningRate = 0.01;

        // Track memory usage before training
        Runtime runtime = Runtime.getRuntime();
        final long[] maxMemoryUsed = { 0 };

        NeuralNetwork nn = new NeuralNetwork(784, numberOfClasses, FILENAME, testFILENAME, HiddenLayerNumNeurons,
                epochs,
                learningRate, batchSize);
        // Start a thread to monitor memory usage during training
        Thread memoryMonitor = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                long used = runtime.totalMemory() - runtime.freeMemory();
                if (used > maxMemoryUsed[0]) {
                    maxMemoryUsed[0] = used;
                }
                try {
                    Thread.sleep(100); // check every 100ms
                } catch (InterruptedException e) {
                    break;
                }
            }
        });
        memoryMonitor.start();
        long startTime = System.currentTimeMillis();

        nn.train();
        System.out.println("Time: " + (System.currentTimeMillis() - startTime) / 1000.0f);
        System.out.println("Mem: " + maxMemoryUsed[maxMemoryUsed.length - 1] / 1000.0f / 1000 / 1000 + "GB");
        memoryMonitor.interrupt();

    }
}
