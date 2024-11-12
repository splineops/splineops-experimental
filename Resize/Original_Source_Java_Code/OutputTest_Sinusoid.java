import java.io.FileWriter;
import java.io.IOException;

public class OutputTest_Sinusoid {
    public static void main(String[] args) {
        // Create a 5x5 image with a white square in the middle and sinusoidal pattern overlay
        ImageAccess inputImage = new ImageAccess(5, 5);
        ImageAccess upscaledImage = new ImageAccess(10, 10);  // Upscaled to 10x10
        ImageAccess revertedImage = new ImageAccess(5, 5);    // Reverted back to 5x5

        // Parameters for the sinusoidal pattern
        double freqX = 2.0 * Math.PI / 5.0;  // Frequency along x-axis
        double freqY = 3.0 * Math.PI / 5.0;  // Different frequency along y-axis

        // Fill the input image with a white square and a sinusoidal pattern
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                // Generate sinusoidal pattern value based on x and y coordinates
                double sinusoid = 0.5 * (Math.sin(freqX * x) + Math.cos(freqY * y));
                
                // Add the sinusoidal pattern on top of the square
                if (x >= 1 && x < 4 && y >= 1 && y < 4) {
                    inputImage.putPixel(x, y, 1.0 + sinusoid);  // White pixel + sinusoid
                } else {
                    inputImage.putPixel(x, y, sinusoid);  // Sinusoid only outside the square
                }
            }
        }

        // Resize parameters
        int analyDegree = 1;
        int syntheDegree = 3;
        int interpolationDegree = 3;
        double zoomFactorX = 2.0;
        double zoomFactorY = 2.0;
        double shiftY = 0.0;
        double shiftX = 0.0;
        boolean inversable = false;

        // Instantiate the Resize class
        Resize resizer = new Resize();

        // Step 1: Upscale the image to 10x10
        resizer.computeZoom(inputImage, upscaledImage, analyDegree, syntheDegree,
                            interpolationDegree, zoomFactorY, zoomFactorX,
                            shiftY, shiftX, inversable);

        // Step 2: Downscale back to 5x5
        double reverseZoomFactor = 1.0 / zoomFactorX;
        resizer.computeZoom(upscaledImage, revertedImage, analyDegree, syntheDegree,
                            interpolationDegree, reverseZoomFactor, reverseZoomFactor,
                            shiftY, shiftX, inversable);

        // Write upscaled and reverted images to CSV files
        writeImageToCSV(upscaledImage, "java_upscaled_10x10.csv", "Upscaled Image Data");
        writeImageToCSV(revertedImage, "java_reverted_5x5.csv", "Reverted Image Data");
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
