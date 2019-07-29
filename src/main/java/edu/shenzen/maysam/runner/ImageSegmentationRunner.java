package edu.shenzen.maysam.runner;

import edu.shenzen.maysam.entities.math.Point;
import edu.shenzen.maysam.entities.ml.Centroid;
import edu.shenzen.maysam.initial.DataInitialization;
import edu.shenzen.maysam.util.ImageUtils;
import edu.shenzen.maysam.util.ml.ClusteringUtil;
import edu.shenzen.maysam.util.ml.KMeans;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ImageSegmentationRunner {
    private static final String FILE_PATH_BASE="C:\\Users\\Yasin\\Desktop\\108073\\";

    private static final String FILENAME_READ = "105053.jpg";
    private static final String FILENAME_WRITE = "105053_1.jpg";
    private static final String FILENAME_GROUNDTRUTH = "105053.seg";

    public static void main(String[] args) throws Exception{

        Map<Integer, Integer> groundTruthAsMap = readGroundTruth(FILE_PATH_BASE+FILENAME_GROUNDTRUTH);
        Map<Integer, Integer> groundTruthIndex = new HashMap<>();
        int counter = 0;
        for (Map.Entry<Integer, Integer> entry : groundTruthAsMap.entrySet()) {
            groundTruthIndex.put(entry.getKey(), counter++);
        }
        int[] groundTruth = new int[groundTruthAsMap.size()];
        for (Map.Entry<Integer, Integer> entry : groundTruthAsMap.entrySet()) {
            groundTruth[groundTruthIndex.get(entry.getKey())] = entry.getValue();
        }


        String filename = FILE_PATH_BASE + FILENAME_READ;
        BufferedImage bi = ImageIO.read(new File(filename));

        //ImageUtils.showImage(filename);

        List<Point> dataset = DataInitialization.createDataset(filename);

        KMeans kMeans = new KMeans(dataset);
        List<Centroid> centroids = kMeans.performClustering(10);

        int[] solutionDS = new int[groundTruthAsMap.size()];
        int cluster = 0;
        for (Centroid entry : centroids) {
            for (Integer index : entry.getTrainingSamples().keySet()) {
                if(groundTruthAsMap.containsKey(index)){
                    solutionDS[groundTruthIndex.get(index)]=cluster;
                }
            }
            cluster = cluster+1;
        }

        double rndIndex = ClusteringUtil.randIndex(groundTruth, solutionDS);
        System.out.println(rndIndex);
        /*int[][] modified_red_pixel = new int[bi.getWidth()][bi.getHeight()];
        int[][] modified_green_pixel = new int[bi.getWidth()][bi.getHeight()];
        int[][] modified_blue_pixel = new int[bi.getWidth()][bi.getHeight()];

        Color[] colors = new Color[]{Color.BLACK, Color.BLUE, Color.CYAN, Color.RED, Color.GREEN, Color.YELLOW,
        Color.MAGENTA, Color.ORANGE, Color.PINK, Color.WHITE};
        int index = 0;
        for (Centroid centroid : centroids) {
            Point features = centroid.getFeatures();
            Map<Integer, Point> trainingSamples = centroid.getTrainingSamples();
            for (Map.Entry<Integer, Point> e : trainingSamples.entrySet()) {
                int x = e.getKey() % bi.getWidth();
                int y = e.getKey() / bi.getWidth();
                modified_red_pixel[x][y] = colors[index].getRed();//new Double(features.getCoords()[0]).intValue();
                modified_green_pixel[x][y] = colors[index].getGreen();//new Double(features.getCoords()[1]).intValue();
                modified_blue_pixel[x][y] = colors[index].getBlue();//new Double(features.getCoords()[2]).intValue();
            }
            index++;
        }*/



//        String newFileName = "C:\\Users\\Yasin\\Desktop\\download_2.jpg";

  //      ImageUtils.writeRGB(modified_red_pixel, modified_green_pixel, modified_blue_pixel, newFileName);

    }

    private static Map<Integer, Integer> readGroundTruth(String filename) throws Exception{
        List<String> lines = Files.lines(Paths.get(filename)).collect(Collectors.toList());

        int width = Integer.parseInt(lines.get(4).split(" ")[1]);
        int height = Integer.parseInt(lines.get(5).split(" ")[1]);
        int[][] picture = new int[height][width];
        for(int i = 0 ; i < picture.length ; i++)
            for(int j = 0 ; j < picture[i].length ; j++)
                picture[i][j] = -1;

        Map<Integer,Integer> result = new HashMap<>();

        for(int i = 11 ; i < lines.size() ; i++){
            int x = Integer.parseInt(lines.get(i).split(" ")[1]);
            int y1 = Integer.parseInt(lines.get(i).split(" ")[2]);
            int y2 = Integer.parseInt(lines.get(i).split(" ")[3]);

            picture[x][y1] = Integer.parseInt(lines.get(i).split(" ")[0]);
            picture[x][y2] = Integer.parseInt(lines.get(i).split(" ")[0]);
        }
        int index = 0;
        for(int i = 0 ; i < picture.length ; i++) {
            for (int j = 0; j < picture[i].length; j++) {
                if(picture[i][j] != -1){
                    result.put(index, picture[i][j]);
                }
                index++;
            }
        }
        return result;
    }

}
