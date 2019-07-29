package edu.shenzen.maysam.entities.comparators;

import edu.shenzen.maysam.entities.solutions.KMeansSolution;

import java.util.Comparator;

public class SimpleSolutionComparator implements Comparator<KMeansSolution> {

    public SimpleSolutionComparator() {

    }

    @Override
    public int compare(KMeansSolution o1, KMeansSolution o2) {
     return new Double(o1.getDb()).compareTo(o2.getDb());
    }

}
