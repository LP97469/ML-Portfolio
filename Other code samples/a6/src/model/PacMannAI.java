package model;

import java.util.Random;
import model.GameModel.Item;
import model.MazeGraph.Direction;
import model.MazeGraph.IPair;
import model.MazeGraph.MazeEdge;
import model.MazeGraph.MazeVertex;
import util.GameMap;
import util.MazeGenerator.TileType;


/**
 * Ramo, Klevis. “Convolution.java.” Java Machine Learning for Computer Vision, GitHub, 25 July 2018,
 * https://github.com/PacktPublishing/Java-Machine-Learning-for-Computer-Vision/blob/master/EdgeDetection/src/main/java/ramo/klevis/ml/Convolution.java.
 * Accessed 6 May 2025.
 */
class Convolution {

    public static double singlePixelConvolution(double[][] input,
            int x, int y,
            double[][] k,
            int kernelWidth,
            int kernelHeight) {
        double output = 0;
        for (int i = 0; i < kernelWidth; ++i) {
            for (int j = 0; j < kernelHeight; ++j) {
                output = output + (input[x + i][y + j] * k[i][j]);
            }
        }
        return output;
    }

    public static double[][] convolution2D(double[][] input,
            int width, int height,
            double[][] kernel,
            int kernelWidth,
            int kernelHeight) {
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        double[][] output = new double[smallWidth][smallHeight];
        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output[i][j] = 0;
            }
        }
        for (int i = 0; i < smallWidth; ++i) {
            for (int j = 0; j < smallHeight; ++j) {
                output[i][j] = singlePixelConvolution(input, i, j, kernel,
                        kernelWidth, kernelHeight);
            }
        }
        return output;
    }

    public static double[][] convolution2DPadded(double[][] input,
            int width, int height,
            double[][] kernel,
            int kernelWidth,
            int kernelHeight) {
        int smallWidth = width - kernelWidth + 1;
        int smallHeight = height - kernelHeight + 1;
        int top = kernelHeight / 2;
        int left = kernelWidth / 2;

        double[][] small = convolution2D(input, width, height,
                kernel, kernelWidth, kernelHeight);
        double large[][] = new double[width][height];
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                large[i][j] = 0;
            }
        }
        for (int j = 0; j < smallHeight; ++j) {
            for (int i = 0; i < smallWidth; ++i) {
                large[i + left][j + top] = small[i][j];
            }
        }
        return large;
    }

    public static double[][][] convolution3D(
            double[][][] input,
            int width, int height,
            double[][][] kernels,
            int kernelWidth, int kernelHeight) {

        int channels = input.length;
        // output spatial dims
        int outW = width  - kernelWidth  + 1;
        int outH = height - kernelHeight + 1;

        double[][][] output = new double[channels][outW][outH];

        for (int c = 0; c < channels; c++) {
            // for each channel, just call your 2D conv:
            output[c] = convolution2DPadded(
                    input[c],
                    width, height,
                    kernels[c],
                    kernelWidth, kernelHeight
            );
        }

        return output;
    }

}


/**
 * “Java Program to Multiply two Matrices of any size.” GeeksforGeeks, 4 July 2024,
 * https://www.geeksforgeeks.org/java-program-to-multiply-two-matrices-of-any-size/.
 * Accessed 6 May 2025.
 */
class MMult {
    public static double[] multiplyMatrixVector(double[][] A, double[] v) {
        int m = A.length;
        if (m == 0) return new double[0];
        int n = A[0].length;
        if (v.length != n) {
            throw new IllegalArgumentException(
                    "Matrix columns (" + n + ") must equal vector length (" + v.length + ")");
        }

        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += A[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }
}


class RELU{

    public static double[] oneDRelu(double[] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = (matrix[i] >= 0)
                    ? matrix[i]
                    : 0;

        }
        return matrix;
    }

    public static double[][][] threeDRelu(double[][][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                for (int k = 0; k < matrix[i][j].length; j++) {
                    matrix[i][j][k] = (matrix[i][j][k] >= 0)
                            ? matrix[i][j][k]
                            : 0;

                }
            }
        }
        return matrix;
    }

}

class Pooling {

    public static double[] globalAvgPool(double[][][] input) {
        int channels = input.length;
        if (channels == 0) return new double[0];

        int height = input[0].length;
        int width  = input[0][0].length;
        double[] output = new double[channels];

        for (int c = 0; c < channels; c++) {
            double sum = 0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += input[c][i][j];
                }
            }
            output[c] = sum / (height * width);
        }

        return output;
    }
}




public class PacMannAI extends PacMann{

    /**
     * Random number generator
     */
    private final Random random = new Random();

    /**
     * mapping from index to actions
     */
    private static final Direction[] actions = Direction.values();

    /**
     * map
     */
    private final GameMap map;

    // channel indices
    public static final int C_PATH    = 0;
    public static final int C_PELLET  = 1;
    public static final int C_BLINKY  = 2;
    public static final int C_PINKY   = 3;
    public static final int C_INKY    = 4;
    public static final int C_CLYDE   = 5;
    public static final int C_PACMANN = 6;
    public static final int CHANNELS  = 7;

    double[][][] stateTensor;

    //actor locations
    private IPair blinkyloc;
    private IPair pinkyloc;
    private IPair inkyloc;
    private IPair clydeloc;
    private IPair pacmannloc;

    /**
     * score
     */
    private int score;






    /**
     * Construct a PacMann character associated to the given `model`.
     *
     * @param model
     */
    public PacMannAI(GameModel model) {
        super(model);
        map = model.map();


        int width = map.types().length;
        int height = map.types()[0].length;
        stateTensor = new double[CHANNELS][height][width];

        // fill path & pellet channels in one go
        TileType[][] types = map.types();
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                stateTensor[C_PATH][row][col]   = (types[col][row] == TileType.PATH)                   ? 1 : 0;
                stateTensor[C_PELLET][row][col] = (model.itemAt(model.graph().closestTo(col, row)) != Item.NONE) ? 1 : 0;
            }
        }




        // make an array for items

    }

    public void updateActorPositions(){
        if (blinkyloc != null){
            stateTensor[C_BLINKY][blinkyloc.j()][blinkyloc.i()] = 0;
        }
        blinkyloc   = model.blinky().nearestVertex().loc();
        stateTensor[C_BLINKY][blinkyloc.j()][blinkyloc.i()] = 1;

        if (pinkyloc != null){
            stateTensor[C_PINKY][pinkyloc.j()][pinkyloc.i()] = 0;
        }
        pinkyloc    = model.pinky().nearestVertex().loc();
        stateTensor[C_PINKY][pinkyloc.j()][pinkyloc.i()]   = 1;

        if (inkyloc != null){
            stateTensor[C_INKY][inkyloc.j()][inkyloc.i()] = 0;
        }
        inkyloc     = model.inky().nearestVertex().loc();
        stateTensor[C_INKY][inkyloc.j()][inkyloc.i()]      = 1;

        if (clydeloc != null){
            stateTensor[C_CLYDE][clydeloc.j()][clydeloc.i()] = 0;
        }
        clydeloc    = model.clyde().nearestVertex().loc();
        stateTensor[C_CLYDE][clydeloc.j()][clydeloc.i()]    = 1;

        if (pacmannloc != null){
            stateTensor[C_PACMANN][pacmannloc.j()][pacmannloc.i()] = 0;
        }
        pacmannloc  = model.pacMann().nearestVertex().loc();
        stateTensor[C_PACMANN][pacmannloc.j()][pacmannloc.i()] = 1;
    }

    /**
     * Returns the next edge that this actor will traverse in the game graph. Navigation strategy is
     * delegated to the subclass.  Will only be called when this actor is standing on a vertex,
     * which must equal the returned edge's starting vertex.
     */
    @Override
    public MazeEdge nextEdge() {
        // rng baseline is about 150 score
        //MazeVertex v = model.pacMann().nearestVertex();
        //Direction d = actions[random.nextInt(3)];
        //return v.edgeInDirection(d);

        updateActorPositions();

        System.out.println(stateTensor);

        MazeVertex v = model.pacMann().nearestVertex();
        Direction d = actions[random.nextInt(3)];
        return v.edgeInDirection(d);

        // need to make zero arrays for all the ghosts
        // need to make zero array for the self


        //model.itemAt()

        //return null;
    }
}
