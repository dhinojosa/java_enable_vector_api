package com.evolutionnext.vector;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class LinearRegression {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private double intercept;
    private double slope;
    private boolean trained = false;

    public void fit(double[] x, double[] y) {
        if (x.length != y.length)
            throw new IllegalArgumentException("x and y must have the same length");

        int length = x.length;
        double xSum = vectorSum(x);
        double ySum = vectorSum(y);
        double xMean = xSum / length;
        double yMean = ySum / length;

        double numerator = 0.0;
        double denominator = 0.0;

        for (int i = 0; i < length; i += SPECIES.length()) {
            VectorMask<Double> mask = SPECIES.indexInRange(i, length);
            DoubleVector vx = DoubleVector.fromArray(SPECIES, x, i, mask);
            DoubleVector vy = DoubleVector.fromArray(SPECIES, y, i, mask);

            DoubleVector xCentered = vx.sub(DoubleVector.broadcast(SPECIES, xMean));
            DoubleVector yCentered = vy.sub(DoubleVector.broadcast(SPECIES, yMean));

            DoubleVector num = xCentered.mul(yCentered);
            DoubleVector den = xCentered.mul(xCentered);

            numerator += num.reduceLanes(VectorOperators.ADD, mask);
            denominator += den.reduceLanes(VectorOperators.ADD, mask);
        }

        this.slope = numerator / denominator;
        this.intercept = yMean - this.slope * xMean;
        this.trained = true;
    }

    public double predict(double x) {
        ensureTrained();
        return intercept + slope * x;
    }

    public double[] predict(double[] x) {
        ensureTrained();
        double[] results = new double[x.length];
        for (int i = 0; i < x.length; i += SPECIES.length()) {
            VectorMask<Double> mask = SPECIES.indexInRange(i, x.length);
            DoubleVector vx = DoubleVector.fromArray(SPECIES, x, i, mask);
            DoubleVector result = vx.mul(slope).add(intercept);
            result.intoArray(results, i, mask);
        }
        return results;
    }

    public double getIntercept() {
        ensureTrained();
        return intercept;
    }

    public double getSlope() {
        ensureTrained();
        return slope;
    }

    private void ensureTrained() {
        if (!trained)
            throw new IllegalStateException("Model must be trained first. Call fit(x, y).");
    }

    private double vectorSum(double[] arr) {
        double sum = 0.0;
        for (int i = 0; i < arr.length; i += SPECIES.length()) {
            VectorMask<Double> mask = SPECIES.indexInRange(i, arr.length);
            DoubleVector v = DoubleVector.fromArray(SPECIES, arr, i, mask);
            sum += v.reduceLanes(VectorOperators.ADD, mask);
        }
        return sum;
    }

    public static void main(String[] args) {
        double[] x = {1, 2, 3, 4, 5};
        double[] y = {2.1, 4.0, 6.2, 8.1, 10.1};

        LinearRegression model = new LinearRegression();
        model.fit(x, y);

        System.out.printf("Slope: %.4f%n", model.getSlope());
        System.out.printf("Intercept: %.4f%n", model.getIntercept());

        double prediction = model.predict(3.2);
        System.out.printf("Predicted y for x = 3.2: %.4f%n", prediction);

        double[] xNew = {6.0, 7.0, 8.0};
        double[] yPred = model.predict(xNew);
        System.out.println("Batch predictions:");
        for (double yp : yPred)
            System.out.printf("%.4f ", yp);
    }
}
