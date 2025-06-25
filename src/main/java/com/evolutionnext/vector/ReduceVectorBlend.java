package com.evolutionnext.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class ReduceVectorBlend {
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static double vectorComputation(float[] a, float threshold) {
        float sum = 0f;
        for (int i = 0; i < a.length; i += SPECIES.length()) {
            VectorMask<Float> mask = SPECIES.indexInRange(i, a.length);
            FloatVector va = FloatVector.fromArray(SPECIES, a, i, mask);
            VectorMask<Float> cond = va.compare(VectorOperators.GT, threshold, mask);
            FloatVector squared = va.mul(va, mask);

            // Blend is ternary: if cond is true, then squared else 0
            FloatVector result = FloatVector.zero(SPECIES).blend(squared, cond);

            // Sum active lanes only
            sum += result.reduceLanes(VectorOperators.ADD, cond);
        }

        return sum;
    }

    public static void main(String[] args) {
        float[] a = {0.23f, 0.45f, 0.49f, 0.90f, 0.01f, 0.88f, 0.79f, 0.45f, 0.21f, 0.93f};
        System.out.printf("a length %d%n", a.length);
        double sumOfSquaresOver50percent = vectorComputation(a, 0.5f);
        System.out.printf("Conditional Sum of Squares (x > 0.5): %f%n", sumOfSquaresOver50percent);

        double sumOfSquaresOver90percent = vectorComputation(a, 0.9f);
        System.out.printf("Conditional Sum of Squares (x > 0.9): %f%n", sumOfSquaresOver90percent);
    }
}
