import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class OutputTest {
    public static void main(String[] args) throws IOException {
        // Load input image from CSV
        ImageAccess inputImage = loadImageFromCSV("input_image.csv");
        ImageAccess outputImage = new ImageAccess(250, 250);  // Assuming output size with 0.5 zoom

        // Resize parameters
        int analyDegree = 3;
        int syntheDegree = 3;
        int interpolationDegree = 3;
        double zoomFactorX = 0.5;
        double zoomFactorY = 0.5;
        double shiftY = 0.0;
        double shiftX = 0.0;
        boolean inversable = false;

        // Instantiate the Resize class
        Resize resizer = new Resize();
        
        // Perform resizing operation using computeZoom
        resizer.computeZoom(inputImage, outputImage, analyDegree, syntheDegree, 
                            interpolationDegree, zoomFactorY, zoomFactorX, 
                            shiftY, shiftX, inversable);
        
        // Write resized image data to a CSV file for Python to read
        try (FileWriter csvWriter = new FileWriter("java_output.csv")) {
            csvWriter.append("Resized Image Data\n");
            for (int y = 0; y < outputImage.getHeight(); y++) {
                for (int x = 0; x < outputImage.getWidth(); x++) {
                    csvWriter.append(String.valueOf(outputImage.getPixel(x, y))).append(",");
                }
                csvWriter.append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to load an image from CSV into ImageAccess
    private static ImageAccess loadImageFromCSV(String filepath) throws IOException {
        ArrayList<double[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            br.readLine();  // Skip header
            while ((line = br.readLine()) != null) {
                String[] pixels = line.split(",");
                double[] row = new double[pixels.length];
                for (int i = 0; i < pixels.length; i++) {
                    row[i] = Double.parseDouble(pixels[i]);
                }
                rows.add(row);
            }
        }
        
        int height = rows.size();
        int width = rows.get(0).length;
        ImageAccess image = new ImageAccess(width, height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.putPixel(x, y, rows.get(y)[x]);
            }
        }
        return image;
    }
}
