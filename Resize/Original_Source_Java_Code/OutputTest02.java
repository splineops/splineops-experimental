import java.io.FileWriter;
import java.io.IOException;

public class OutputTest02 {
    public static void main(String[] args) {
        // Create a 10x10 black image with a 5x5 white square in the middle
        ImageAccess inputImage = new ImageAccess(10, 10);
        ImageAccess downscaledImage = new ImageAccess(5, 5);  // Downscaled to 5x5
        ImageAccess revertedImage = new ImageAccess(10, 10);  // Reverted back to 10x10

        // Fill the input image with a white square
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                if (x >= 3 && x < 7 && y >= 3 && y < 7) {
                    inputImage.putPixel(x, y, 1.0);  // White pixel, normalized
                } else {
                    inputImage.putPixel(x, y, 0.0);  // Black pixel
                }
            }
        }

        // Resize parameters
        int analyDegree = 0;
        int syntheDegree = 1;
        int interpolationDegree = 1;
        double zoomFactorX = 0.5;
        double zoomFactorY = 0.5;
        double shiftY = 0.0;
        double shiftX = 0.0;
        boolean inversable = false;

        // Instantiate the Resize class
        Resize resizer = new Resize();

        // Step 1: Downscale the image to 5x5
        resizer.computeZoom(inputImage, downscaledImage, analyDegree, syntheDegree,
                            interpolationDegree, zoomFactorY, zoomFactorX,
                            shiftY, shiftX, inversable);

        // Step 2: Upscale back to 10x10
        double reverseZoomFactor = 1.0 / zoomFactorX;
        resizer.computeZoom(downscaledImage, revertedImage, analyDegree, syntheDegree,
                            interpolationDegree, reverseZoomFactor, reverseZoomFactor,
                            shiftY, shiftX, inversable);

        // Write downscaled and reverted images to CSV files
        writeImageToCSV(downscaledImage, "java_downscaled_5x5.csv", "Downscaled Image Data");
        writeImageToCSV(revertedImage, "java_reverted_10x10.csv", "Reverted Image Data");
    }

    // Method to write ImageAccess data to CSV
    private static void writeImageToCSV(ImageAccess image, String filename, String header) {
        try (FileWriter csvWriter = new FileWriter(filename)) {
            csvWriter.append(header).append("\n");
            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    csvWriter.append(String.valueOf(image.getPixel(x, y))).append(",");
                }
                csvWriter.append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
