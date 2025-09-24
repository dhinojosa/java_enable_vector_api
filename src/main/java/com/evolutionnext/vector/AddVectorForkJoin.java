package com.evolutionnext.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ThreadLocalRandom;

public class AddVectorForkJoin {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int SIZE = 10_000;
    private static final int THRESHOLD = Math.max(SPECIES.length(), SIZE / Runtime.getRuntime().availableProcessors());

    public static void main(String[] args) {

        float[] a = new float[SIZE];
        float[] b = new float[SIZE];
        float[] c = new float[SIZE];

        ThreadLocalRandom rand = ThreadLocalRandom.current();
        for (int i = 0; i < SIZE; i++) {
            a[i] = rand.nextFloat() * 100.0f;
            b[i] = rand.nextFloat() * 100.0f;
        }

        System.out.printf("a length %d%n", a.length);
        System.out.printf("b length %d%n", b.length);
        System.out.printf("c length %d%n", c.length);
        System.out.printf("Preferred SIMD species length = %d floats%n", SPECIES.length());
        System.out.printf("Total bits = %d%n", SPECIES.length() * Float.SIZE);

        try (ForkJoinPool forkJoinPool = ForkJoinPool.commonPool()) {
            forkJoinPool.invoke(new VectorAdditionTask(a, b, c, 0, a.length));
            System.out.println(Arrays.toString(c));
        }
    }

    static class VectorAdditionTask extends RecursiveAction {
        private final float[] a, b, c;
        private final int start, end;

        public VectorAdditionTask(float[] a, float[] b, float[] c, int start, int end) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int length = end - start;
            if (length <= THRESHOLD) {
                computeDirectly();
            } else {
                int mid = start + (length / 2);
                invokeAll(
                    new VectorAdditionTask(a, b, c, start, mid),
                    new VectorAdditionTask(a, b, c, mid, end)
                );
            }
        }

        private void computeDirectly() {
            System.out.printf("Thread %s processing [%d,%d)%n",
                Thread.currentThread().getName(), start, end);
            for (int index = start; index < end; index += SPECIES.length()) {
                var mask = SPECIES.indexInRange(index, end);
                var va = FloatVector.fromArray(SPECIES, a, index, mask);
                var vb = FloatVector.fromArray(SPECIES, b, index, mask);
                var vc = va.add(vb, mask);
                vc.intoArray(c, index, mask);
            }
        }
    }
}
