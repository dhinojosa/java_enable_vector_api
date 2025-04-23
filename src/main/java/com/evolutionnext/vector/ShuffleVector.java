package com.evolutionnext.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

public class ShuffleVector {
    static final VectorSpecies<Float> SPECIES =
        FloatVector.SPECIES_PREFERRED;

    static void reverseWithShuffle(float[] input, float[] output) {
        int length = input.length;
        for (int i = 0; i < length; i += SPECIES.length()) {
            var mask = SPECIES.indexInRange(i, length);
            var vector = FloatVector.fromArray(SPECIES, input, i, mask);

            // Reversed shuffle: last-to-first
            int lanes = SPECIES.length();
            int[] reversedIndexes = new int[lanes];
            for (int j = 0; j < lanes; j++) reversedIndexes[j] = lanes - 1 - j;
            var shuffle = VectorShuffle.fromArray(SPECIES, reversedIndexes, 0);
            var reversed = vector.rearrange(shuffle, mask);

            reversed.intoArray(output, i, mask);
        }
    }

    public static void main(String[] args) {
        float[] a = {10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f};
        float[] result = new float[a.length];

        reverseWithShuffle(a, result);

        System.out.println("Original: " + Arrays.toString(a));
        System.out.println("Reversed (per block): " + Arrays.toString(result));
    }
}
