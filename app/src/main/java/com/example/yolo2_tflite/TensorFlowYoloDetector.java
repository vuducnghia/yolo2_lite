package com.example.yolo2_tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;


import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;


import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

public class TensorFlowYoloDetector implements Classifier {
    private static final int MAX_RESULTS = 25;

    private final static float OVERLAP_THRESHOLD = 0.2f;

    private static final int NUM_CLASSES = 1;

    private static final int NUM_BOXES_PER_BLOCK = 5;

    private static final float CONFIDENT_IN_CLASS = 0.25f;

    private static final double[] ANCHORS = {
            0.57273, 0.677385, 0.87446, 1.06253, 3.33843, 5.47434, 5.88282, 3.52778, 7.77052, 7.16828
    };

    private static final String[] LABELS = {
            "face"
    };
    private final float[][][][] outputNet = new float[1][13][13][30];

    protected ByteBuffer imgData = null;

    private int inputSize;

    private int[] intValues;
    private int blockSize;

    private boolean logStats = false;

    protected Interpreter tflite;

//    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    GpuDelegate gpuDelegate = new GpuDelegate();

    Interpreter.Options options = (new Interpreter.Options()).addDelegate(gpuDelegate);

    /**
     * Initializes a native TensorFlow session for classifying images.
     */
    public static Classifier create(final AssetManager assetManager, final String modelFilename,
                                    final int inputSize, final int blockSize) throws IOException {

        TensorFlowYoloDetector d = new TensorFlowYoloDetector();
        d.inputSize = inputSize;
        d.intValues = new int[inputSize * inputSize];
        d.blockSize = blockSize;
        MappedByteBuffer tfliteModel = loadModelFile(assetManager, modelFilename);
        d.options.setNumThreads(25);
        d.tflite = new Interpreter(tfliteModel, d.options);


        return d;
    }

    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String inputName) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(inputName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TensorFlowYoloDetector() {
    }

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < 416; ++i) {
            for (int j = 0; j < 416; ++j) {
                final int val = intValues[pixel++];
                addPixelValue(val);
            }
        }
    }

    protected void addPixelValue(int pixelValue) {
        imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
        imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
        imgData.putFloat((pixelValue & 0xFF) / 255.0f);
    }


    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        //1 * 416 * 416 * 3 * 4
        imgData = ByteBuffer.allocateDirect(2076672);

        imgData.order(ByteOrder.nativeOrder());
        long startTime = SystemClock.uptimeMillis();

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());


        final int gridWidth = 13; // bitmap.getWidth() / blockSize;
        final int gridHeight = 13; //bitmap.getHeight() / blockSize;
        final float[] output = new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];

        convertBitmapToByteBuffer(bitmap);
        tflite.run(imgData, outputNet);

        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset = (NUM_CLASSES + 5) * b;
                    final float xPos = (x + expit(outputNet[0][y][x][offset])) * blockSize;
                    final float yPos = (y + expit(outputNet[0][y][x][offset + 1])) * blockSize;

                    final float w = (float) (Math.exp(outputNet[0][y][x][offset + 2]) * ANCHORS[2 * b]) * blockSize;
                    final float h = (float) (Math.exp(outputNet[0][y][x][offset + 3]) * ANCHORS[2 * b + 1]) * blockSize;

                    final RectF rect =
                            new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                    Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                    final float confidence = expit(outputNet[0][y][x][offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    classes[0] = outputNet[0][y][x][offset + 5];
                    softmax(classes);

                    if (classes[0] > maxClass) {
                        detectedClass = 0;
                        maxClass = classes[0];
                    }

                    final float confidenceInClass = maxClass * confidence;

                    if (confidenceInClass > CONFIDENT_IN_CLASS) {

                        pq.add(new Recognition("id_" + offset, LABELS[detectedClass], confidenceInClass, rect));
                    }
                }
            }
        }


        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();

        if (pq.size() > 0) {
            // get recognition that has max confident
            recognitions.add(pq.poll());

            for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
                Recognition recognition = pq.poll();
                boolean overlaps = false;

                for (Recognition previousRecognition : recognitions) {
                    float overLap = getIntersectionProportion(previousRecognition.getLocation(), recognition.getLocation());
                    overlaps = overlaps || (overLap > OVERLAP_THRESHOLD);
                    if (overlaps) {
                        break;
                    }
                }

                if (!overlaps) {
                    recognitions.add(recognition);
                }
            }
        }


        long endTime = SystemClock.uptimeMillis();
        System.out.println("Timecost---" + Long.toString(endTime - startTime));

        return recognitions;
    }


    private float getIntersectionProportion(RectF primaryShape, RectF secondaryShape) {
        if (overlaps(primaryShape, secondaryShape)) {
            float intersectionSurface = Math.max(0, Math.min(primaryShape.right, secondaryShape.right) - Math.max(primaryShape.left, secondaryShape.left)) *
                    Math.max(0, Math.min(primaryShape.bottom, secondaryShape.bottom) - Math.max(primaryShape.top, secondaryShape.top));

            float surfacePrimary = Math.abs(primaryShape.right - primaryShape.left) * Math.abs(primaryShape.bottom - primaryShape.top);

            return intersectionSurface / surfacePrimary;
        }

        return 0f;
    }

    private boolean overlaps(RectF primary, RectF secondary) {
        return primary.left < secondary.right && primary.right > secondary.left
                && primary.top < secondary.bottom && primary.bottom > secondary.top;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return tflite.toString();
    }

    @Override
    public void close() {
        tflite.close();
    }
}
