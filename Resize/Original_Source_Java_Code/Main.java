public class Main {
    public static void main(String[] args) {
        // Create a 10x10 black image with a 5x5 white square in the middle
        ImageAccess input = new ImageAccess(10, 10);
        ImageAccess output = new ImageAccess(5, 5);

        // Fill the input image with black and a white square in the center
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                if (x >= 2 && x < 7 && y >= 2 && y < 7) {
                    input.putPixel(x, y, 1.0);  // White pixel
                } else {
                    input.putPixel(x, y, 0.0);  // Black pixel
                }
            }
        }

        // Create an instance of Resize and call computeZoom
        Resize resizer = new Resize();

        // Parameters for computeZoom
        // input        : ImageAccess object to be resized
        // output       : ImageAccess object which is the resized version of the input
        // analyDegree  : Degree of the analysis spline (3 in this case)
        // syntheDegree : Degree of the synthesis spline (3 in this case)
        // interpDegree : Degree of the interpolating spline (3 in this case)
        // zoomY        : Scaling factor that applies to the columns (0.5 to reduce by half)
        // zoomX        : Scaling factor that applies to the rows (0.5 to reduce by half)
        // shiftY       : Shift value that applies to the columns (0 in this case)
        // shiftX       : Shift value that applies to the rows (0 in this case)
        // inversable   : Boolean indicating if the image is inversable (false in this case)
        resizer.computeZoom(input, output, 3, 3, 3, 0.5, 0.5, 0, 0, false);

        // Print the resized image values
        System.out.println("Resized Image (5x5):");
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                System.out.print(output.getPixel(x, y) + " ");
            }
            System.out.println();
        }
    }
}

/*
 * 
Resized Image using Custom Method (5x5):
0.04 -0.17 -0.21 -0.16 0.01 
-0.17 0.70 0.88 0.64 -0.04
-0.21 0.88 1.10 0.80 -0.05
-0.16 0.64 0.80 0.59 -0.04
0.01 -0.04 -0.05 -0.04 0.00

--------------------------------------------------

Ground Truth Image (5x5):
0.04 -0.17 -0.21 -0.16 0.01
-0.17 0.70 0.88 0.64 -0.04
-0.21 0.88 1.10 0.80 -0.05
-0.16 0.64 0.80 0.59 -0.04
0.01 -0.04 -0.05 -0.04 0.00

--------------------------------------------------
 * 
 */
