package com.evolutionnext.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

public class BroadcastVector {
    static final VectorSpecies<Float> SPECIES =
        FloatVector.SPECIES_PREFERRED;

    static void vectorComputation(float[] a, float[] result) {
        for (int index = 0; index < a.length; index += SPECIES.length()) {
            var mask = SPECIES.indexInRange(index, a.length);
            var va = FloatVector.fromArray(SPECIES, a, index, mask);
            var zero = FloatVector.broadcast(SPECIES, 0.0f);
            va.max(zero).intoArray(result, index, mask);
        }
    }

    public static void main(String[] args) {
        float[] a = {0.23f, 0.45f, 0.49f, 0.0f, -0.01f, 0.88f, 0.79f, 0.45f, -0.21f, 0.93f};
        float[] result = new float[a.length];
        vectorComputation(a, result);
        System.out.println("ReLU is: " + Arrays.toString(result));
    }
}
